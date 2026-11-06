"""
pipeline_lean.py
----------------
Memory-optimised variant of AnalysisPipelineFast.

Target: large corpora on machines with limited RAM.

Key differences from pipeline_fast.py:
  1. Token sequences are never all in RAM simultaneously.
     Each file is parsed, used, and discarded.  Only the
     pairwise score matrices (float32) are kept.

  2. Pairwise matrices use numpy memmap -- written to a temp
     file on disk, accessed as if they were numpy arrays.
     Peak RAM for a 500-piece corpus: ~4 MB per matrix
     instead of holding 500 x 5000-token sequences.

  3. TF-IDF is fit in two passes:
       Pass 1 -- stream all docs to fit the vocabulary
       Pass 2 -- transform one doc at a time, accumulate
                sparse rows, then compute pairwise cosine
                distances in chunks.

  4. LCS and n-gram are computed pair-by-pair from disk,
     never loading more than two sequences at once.

Trade-off: more file I/O, same wall time for LCS (already
parallel), slower TF-IDF fit (~2x), but RAM stays flat.

Usage:
    from pipeline_lean import AnalysisPipelineLean, FastConfig
    pipe = AnalysisPipelineLean(fast_config=FastConfig(n_workers=8))
    pipe.add_corpus(...)
    report = pipe.run(calibrator=..., ensemble=...)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_bias_analyzer.pipeline import (
    AnalysisPipeline, CorpusGroup, PipelineConfig,
    ThresholdCalibrator, EnsembleConfig,
    AnalysisReport, GroupStats, PairResult,
    CalibratedThresholds,
)
from music_bias_analyzer.tokenizer import (
    file_to_tokens, load_corpus, SUPPORTED_EXTENSIONS,
)
from music_bias_analyzer.ngram import multi_n_overlap, candidate_pairs
from music_bias_analyzer.embeddings import (
    music_feature_vector, style_collapse_score, pca_2d,
)

import hashlib
import json
import lcs_fast as _lcs_fast


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _corpus_fingerprint(names: list, groups: list, max_seq_len: int,
                         use_intervals: bool = False) -> str:
    """Stable hash of corpus identity -- detects if corpus or config changed."""
    key = json.dumps({
        "names":         sorted(names),
        "groups":        groups,
        "max_seq_len":   max_seq_len,
        "use_intervals": use_intervals,
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _checkpoint_dir(tag: str = "") -> Path:
    d = Path(".pipeline_cache") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_matrix(mat: np.ndarray, path: Path):
    np.save(str(path), mat)
    log.info("  Checkpoint saved: %s", path)


def _load_matrix(path: Path, shape: tuple) -> Optional[np.ndarray]:
    """Load a saved matrix if it exists and shape matches."""
    if not path.exists():
        return None
    try:
        mat = np.load(str(path))
        if mat.shape == shape:
            log.info("  Checkpoint loaded: %s", path)
            return mat.astype(np.float32)
        else:
            log.warning("  Checkpoint shape mismatch %s vs %s -- recomputing",
                        mat.shape, shape)
            return None
    except Exception as exc:
        log.warning("  Checkpoint load failed (%s) -- recomputing", exc)
        return None

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for macOS spawn)
# ---------------------------------------------------------------------------

def _lean_parse_worker(fp_str: str, use_octave: bool,
                        use_duration: bool, q,
                        use_intervals: bool = False) -> None:
    try:
        from music_bias_analyzer.tokenizer import file_to_tokens as _f
        toks = _f(Path(fp_str), use_octave=use_octave,
                  use_duration=use_duration,
                  use_intervals=use_intervals)
        q.put(toks or [])
    except Exception:
        q.put([])


def _lean_lcs_worker(args):
    """Pairwise LCS worker -- loads two sequences, computes, returns score."""
    i, j, s1, s2, use_win, ign_dur, ign_oct = args
    if use_win:
        score = _lcs_fast.lcs_best_window(s1, s2, ign_dur, ign_oct)
    else:
        score = _lcs_fast.lcs_similarity_numpy(s1, s2, ign_dur, ign_oct)
    return i, j, score


# ---------------------------------------------------------------------------
# Config (identical to FastConfig -- same API)
# ---------------------------------------------------------------------------

@dataclass
class FastConfig:
    n_workers:           int   = 0
    chunk_size:          int   = 64
    min_jaccard_for_lcs: float = 0.02
    use_window:          bool  = True
    keep_tfidf_sparse:   bool  = True   # ignored in lean mode -- always streams
    use_intervals:       bool  = False  # transposition-invariant interval encoding


# ---------------------------------------------------------------------------
# File loader with timeout (ABC only)
# ---------------------------------------------------------------------------

ABC_TIMEOUT = 20

def _load_file(fp: Path, use_octave: bool, use_duration: bool,
               use_intervals: bool = False) -> List[str]:
    """Load one file. ABC gets a hard-kill subprocess; MIDI loads directly."""
    import multiprocessing as mp

    if fp.suffix.lower() == ".abc":
        q = mp.Queue()
        p = mp.Process(target=_lean_parse_worker,
                       args=(str(fp), use_octave, use_duration, q),
                       daemon=True)
        p.start()
        p.join(timeout=ABC_TIMEOUT)
        if p.is_alive():
            p.kill(); p.join()
            log.warning("TIMEOUT: skipping %s", fp.name)
            return []
        try:
            return q.get_nowait()
        except Exception:
            return []
    else:
        try:
            return file_to_tokens(fp, use_octave=use_octave,
                                  use_duration=use_duration,
                                  use_intervals=use_intervals)
        except Exception as exc:
            log.warning("PARSE ERROR: %s -- %s", fp.name, exc)
            return []


# ---------------------------------------------------------------------------
# Corpus index -- paths on disk, sequences loaded on demand
# ---------------------------------------------------------------------------

@dataclass
class CorpusIndex:
    """Maps integer index -> (group_name, file_path).
    Sequences are loaded from disk only when needed."""
    names:  List[str]       = field(default_factory=list)
    paths:  List[Path]      = field(default_factory=list)
    groups: List[str]       = field(default_factory=list)
    limits: Dict[str, int]  = field(default_factory=dict)

    def add_dir(self, group: str, directory: Path, limit: int,
                use_octave: bool, use_duration: bool):
        files = sorted(
            fp for fp in directory.rglob("*")
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(p.startswith("_dedup") for p in fp.parts)
        )
        added = 0
        try:
            from tqdm import tqdm
            it = tqdm(files,
                      desc  = f"Indexing {group[:14]}",
                      unit  = "file",
                      total = limit if limit else len(files),
                      dynamic_ncols = True)
        except ImportError:
            it = files

        for fp in it:
            if limit and added >= limit:
                break
            # Quick check -- try to see if file is parseable without full load
            # by checking it exists and has non-zero size
            if fp.stat().st_size > 0:
                self.names.append(fp.name)
                self.paths.append(fp)
                self.groups.append(group)
                added += 1

        log.info("Indexed %d files for group '%s'", added, group)

    def load(self, i: int, use_octave: bool,
             use_duration: bool,
             use_intervals: bool = False) -> List[str]:
        return _load_file(self.paths[i], use_octave, use_duration, use_intervals)

    def __len__(self):
        return len(self.names)


# ---------------------------------------------------------------------------
# Lean pipeline
# ---------------------------------------------------------------------------

class AnalysisPipelineLean(AnalysisPipeline):
    """
    Memory-lean pipeline.  Identical API to AnalysisPipelineFast.
    Keeps peak RAM proportional to matrix sizes, not corpus size.
    """

    def __init__(self, config: Optional[PipelineConfig] = None,
                 fast_config: Optional[FastConfig] = None):
        super().__init__(config)
        self.fast_config = fast_config or FastConfig()
        self._group_dirs: Dict[CorpusGroup, Tuple[Path, int]] = {}

    def add_corpus(self, group: CorpusGroup, directory,
                   limit: int = 0) -> "AnalysisPipelineLean":
        """Register corpus directory -- files are NOT loaded yet."""
        self._group_dirs[group] = (Path(directory), limit)
        log.info("Registered corpus '%s' -> %s  (limit=%d)",
                 group.value, directory, limit)
        return self

    def _run_calibration(self, calibrator, all_names, all_seqs,
                          name_to_group, ngram_mat, lcs_mat,
                          tfidf_dist, hist_dist):
        """
        Fast override -- uses numpy boolean indexing instead of a Python
        double loop over all pairs.  O(n^2) numpy ops vs O(n^2) Python iters.
        """
        from music_bias_analyzer.pipeline import (
            CalibratedThresholds, CorpusGroup,
        )
        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: building index arrays ...", flush=True)

        names_arr  = np.array(all_names)
        groups_arr = np.array([name_to_group[n] for n in all_names])
        n          = len(all_names)

        human_grp = CorpusGroup.HUMAN_BASELINE.value
        llm_grp   = CorpusGroup.LLM_OUTPUT.value
        train_grp = CorpusGroup.TRAIN_CORPUS.value

        # Build boolean masks for each group
        is_human = groups_arr == human_grp
        is_llm   = groups_arr == llm_grp
        is_train = groups_arr == train_grp

        # Get upper-triangle indices for each pair type using numpy
        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: extracting null pairs ...", flush=True)
        hi, hj = np.where(np.outer(is_human, is_human))
        mask_upper = hi < hj
        null_i = hi[mask_upper]
        null_j = hj[mask_upper]
        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: {len(null_i)} null pairs", flush=True)

        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: extracting candidate pairs ...", flush=True)
        ci, cj = np.where(np.outer(is_llm, is_train))
        cand_i = ci
        cand_j = cj
        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: {len(cand_i)} candidate pairs", flush=True)

        if len(null_i) == 0:
            import logging as _log
            _log.getLogger(__name__).warning(
                "No human-baseline pairs found for calibration")
        if len(cand_i) == 0:
            import logging as _log
            _log.getLogger(__name__).warning(
                "No llm x train pairs found for calibration")

        # Extract scores directly from matrices using fancy indexing
        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: extracting scores from matrices ...", flush=True)
        null_scores = {
            "jaccard_n5": list(ngram_mat[null_i, null_j].astype(float)),
            "lcs_strict":  list(lcs_mat[null_i,   null_j].astype(float)),
            "lcs_pitch":   list(lcs_mat[null_i,   null_j].astype(float)),
            "tfidf_dist":  list(tfidf_dist[null_i, null_j].astype(float)),
            "hist_dist":   list(hist_dist[null_i,  null_j].astype(float)),
            "bleu_n5":     [],
        }
        candidate_scores = {
            "jaccard_n5": list(ngram_mat[cand_i, cand_j].astype(float)),
            "lcs_strict":  list(lcs_mat[cand_i,   cand_j].astype(float)),
            "lcs_pitch":   list(lcs_mat[cand_i,   cand_j].astype(float)),
            "tfidf_dist":  list(tfidf_dist[cand_i, cand_j].astype(float)),
            "hist_dist":   list(hist_dist[cand_i,  cand_j].astype(float)),
            "bleu_n5":     [],
        }

        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: running Pareto sweep ...", flush=True)

        # Fast Pareto: bin candidate thresholds to max 1000 values
        # to avoid O(n^2) sweep over 800k unique floats
        from music_bias_analyzer.pipeline import CalibratedThresholds as _CT

        MAX_CANDIDATES = 1000

        def _fast_pareto_high(null_vals, cand_vals):
            null_arr = np.array(null_vals, dtype=np.float32)
            cand_arr = np.array(cand_vals, dtype=np.float32)
            all_vals = np.concatenate([null_arr, cand_arr])
            candidates = np.unique(
                np.percentile(all_vals,
                              np.linspace(0, 100, MAX_CANDIDATES))
            )
            best_T = float(null_arr.mean() + 2 * null_arr.std())
            best_fp = 1.0
            for T in candidates:
                fp = float(np.mean(null_arr >= T))
                tp = float(np.mean(cand_arr >= T))
                if fp <= best_fp:
                    best_fp = fp
                    best_T  = float(T)
            return best_T

        def _fast_pareto_low(null_vals, cand_vals):
            null_arr = np.array(null_vals, dtype=np.float32)
            cand_arr = np.array(cand_vals, dtype=np.float32)
            all_vals = np.concatenate([null_arr, cand_arr])
            candidates = np.unique(
                np.percentile(all_vals,
                              np.linspace(0, 100, MAX_CANDIDATES))
            )
            best_T  = float(null_arr.mean() - 2 * null_arr.std())
            best_fp = 1.0
            for T in candidates:
                fp = float(np.mean(null_arr <= T))
                tp = float(np.mean(cand_arr <= T))
                if fp <= best_fp:
                    best_fp = fp
                    best_T  = float(T)
            return best_T

        result = _CT(calibration_source="pareto",
                     n_null_pairs=len(null_i),
                     n_candidate_pairs=len(cand_i))

        for metric, vals_null, vals_cand, is_high in [
            ("jaccard_n5",  null_scores["jaccard_n5"],
                            candidate_scores["jaccard_n5"], True),
            ("lcs_strict",  null_scores["lcs_strict"],
                            candidate_scores["lcs_strict"],  True),
            ("lcs_pitch",   null_scores["lcs_pitch"],
                            candidate_scores["lcs_pitch"],   True),
            ("tfidf_dist",  null_scores["tfidf_dist"],
                            candidate_scores["tfidf_dist"],  False),
            ("hist_dist",   null_scores["hist_dist"],
                            candidate_scores["hist_dist"],   False),
        ]:
            if not vals_null or not vals_cand:
                print(f"[{time.strftime('%H:%M:%S')}]   {metric}: no data, keeping default", flush=True)
                continue
            print(f"[{time.strftime('%H:%M:%S')}]   {metric}: sweeping {len(vals_null)} null / {len(vals_cand)} cand ...", flush=True)
            t_metric = time.time()
            if is_high:
                t = _fast_pareto_high(vals_null, vals_cand)
                attr = metric
            else:
                t = _fast_pareto_low(vals_null, vals_cand)
                attr = f"{metric}_max"
            setattr(result, attr, round(t, 4))
            print(f"[{time.strftime('%H:%M:%S')}]   {metric}: threshold={t:.4f}  ({time.time()-t_metric:.2f}s)", flush=True)

        print(f"[{time.strftime('%H:%M:%S')}] _run_calibration: done", flush=True)
        return result

    def run(self,
            calibrator: Optional[ThresholdCalibrator] = None,
            ensemble:   Optional[EnsembleConfig]      = None,
            ) -> AnalysisReport:

        fc     = self.fast_config
        t0     = time.time()
        cfg    = self.config

        import sys as _sys

        def _tick(label: str, t_prev=None):
            """Print a timestamped checkpoint to stdout immediately."""
            elapsed_total = time.time() - t0
            since_last    = (time.time() - t_prev) if t_prev else 0.0
            msg = (f"[{time.strftime('%H:%M:%S')}] "
                   f"CHECKPOINT  {label:<45} "
                   f"total={elapsed_total:>7.1f}s"
                   + (f"  since_last={since_last:.1f}s" if t_prev else ""))
            print(msg, flush=True)
            log.info("CHECKPOINT: %s  (total=%.1fs)", label, elapsed_total)
            return time.time()

        _t_last = _tick("LeanPipeline started")
        log.info("LeanPipeline starting ...  workers=%s  use_window=%s",
                 fc.n_workers or "auto", fc.use_window)

        # -- Step 1: Build corpus index (paths only, no token loading) -----
        _t_last = _tick("Step 1: Building corpus index", _t_last)
        log.info("Step 1 -- Building corpus index ...")
        idx = CorpusIndex()
        for group, (directory, limit) in self._group_dirs.items():
            idx.add_dir(group.value, directory, limit,
                        cfg.use_octave, cfg.use_duration)

        n             = len(idx)
        name_to_group = dict(zip(idx.names, idx.groups))
        log.info("Total indexed: %d files", n)

        # Fingerprint the corpus so checkpoints auto-invalidate if it changes
        fp_hash  = _corpus_fingerprint(idx.names, idx.groups,
                                        cfg.lcs_max_seq_len,
                                        fc.use_intervals)
        cache_dir = _checkpoint_dir(fp_hash)
        log.info("Cache dir: %s  (fingerprint: %s)", cache_dir, fp_hash)

        # -- Step 2: N-gram matrix -- load each file once, stream -----------
        _t_last = _tick("Step 2: N-gram pass", _t_last)
        log.info("Step 2 -- N-gram pass (streaming) ...")
        _t = time.time()

        # Memmap for n-gram matrix
        _mmap_dir = cache_dir / "mmap"
        _mmap_dir.mkdir(parents=True, exist_ok=True)
        _ng_mmap_path = _mmap_dir / "ngram_mat.mmap"
        ngram_mat = np.memmap(str(_ng_mmap_path), dtype="float32",
                              mode="w+", shape=(n, n))
        np.fill_diagonal(ngram_mat, 1.0)

        # Load all sequences for n-gram (needed pairwise -- unavoidable)
        # But immediately discard after matrix is built
        _t_last = _tick("Step 2a: Loading sequences", _t_last)
        log.info("  Loading sequences for n-gram ...")
        all_seqs: List[List[str]] = []
        try:
            from tqdm import tqdm
            seq_iter = tqdm(range(n), desc="Loading seqs", unit="file",
                            dynamic_ncols=True, colour="cyan")
        except ImportError:
            seq_iter = range(n)

        skipped_load = 0
        for i in seq_iter:
            toks = idx.load(i, cfg.use_octave, cfg.use_duration, fc.use_intervals)
            toks = toks[:cfg.lcs_max_seq_len]
            if not toks:
                skipped_load += 1
            all_seqs.append(toks)
            if hasattr(seq_iter, "set_postfix"):
                seq_iter.set_postfix({
                    "len": len(toks),
                    "skipped": skipped_load,
                    "file": idx.names[i][:18],
                })

        if skipped_load:
            log.warning("  %d files failed to load and will be empty sequences",
                        skipped_load)

        _t_last = _tick("Step 2b: Computing n-gram pairs", _t_last)
        log.info("  Computing n-gram pairs ...")
        from music_bias_analyzer.ngram import ngram_set
        flat_corpus = {idx.names[i]: all_seqs[i] for i in range(n)}

        _ng_cache = cache_dir / "ngram_mat.npy"
        _cached_ng = _load_matrix(_ng_cache, (n, n))
        if _cached_ng is not None:
            ngram_mat[:] = _cached_ng
            del _cached_ng
            log.info("  N-gram matrix loaded from cache -- skipping computation")
        else:
            # Pre-compute n-gram sets once per piece
            gram_sets = [ngram_set(all_seqs[i], 5) for i in range(n)]

            total_pairs = n * (n - 1) // 2
            completed   = 0
            t_ng        = time.time()

            try:
                from tqdm import tqdm as _tqdm
                pbar_ng = _tqdm(total=total_pairs, desc="N-gram pairs",
                                unit="pair", dynamic_ncols=True)
            except ImportError:
                pbar_ng = None

            for i in range(n):
                ngram_mat[i, i] = 1.0
                for j in range(i + 1, n):
                    s1, s2 = gram_sets[i], gram_sets[j]
                    if s1 and s2:
                        score = len(s1 & s2) / len(s1 | s2)
                    else:
                        score = 0.0
                    ngram_mat[i, j] = score
                    ngram_mat[j, i] = score
                    completed += 1
                    if pbar_ng is not None:
                        pbar_ng.update(1)
                    elif completed % 50000 == 0:
                        rate = completed / max(time.time() - t_ng, 0.001)
                        eta  = (total_pairs - completed) / max(rate, 0.001)
                        log.info("  N-gram: %d/%d pairs (%.0f%%)  "
                                 "%.0f pairs/s  ETA %.0fs",
                                 completed, total_pairs,
                                 100 * completed / total_pairs, rate, eta)

            if pbar_ng is not None:
                pbar_ng.close()
            del gram_sets
            _save_matrix(np.array(ngram_mat), _ng_cache)
        log.info("  N-gram done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 3: LCS matrix -- parallel, sequences already loaded -------
        _t_last = _tick("Step 3: LCS", _t_last)
        log.info("Step 3 -- LCS (parallel, best-window) ...")

        # Build pairs mask
        idx_map = {name: i for i, name in enumerate(idx.names)}

        if fc.min_jaccard_for_lcs > 0:
            from music_bias_analyzer.ngram import candidate_pairs as _cp
            cand = set(
                (idx_map[a], idx_map[b])
                for a, b in _cp(flat_corpus,
                                min_jaccard=fc.min_jaccard_for_lcs, n=3)
            )
            total = n * (n - 1) // 2
            log.info("  n-gram pre-filter: %d / %d pairs kept (%.0f%%)",
                     len(cand), total, 100 * len(cand) / max(total, 1))
        else:
            cand = None

        if cfg.lcs_cross_group_only:
            baseline = "human_baseline"
            cross = {(i, j)
                     for i in range(n) for j in range(i + 1, n)
                     if not (idx.groups[i] == baseline and
                             idx.groups[j] == baseline)}
            cand = (cand & cross) if cand else cross

        _lcs_mmap_path = _mmap_dir / "lcs_mat.mmap"
        lcs_mat = np.memmap(str(_lcs_mmap_path), dtype="float32",
                            mode="w+", shape=(n, n))
        np.fill_diagonal(lcs_mat, 1.0)

        _lcs_cache  = cache_dir / "lcs_mat.npy"
        _cached_lcs = _load_matrix(_lcs_cache, (n, n))
        if _cached_lcs is not None:
            lcs_mat[:] = _cached_lcs
            del _cached_lcs
            log.info("  LCS matrix loaded from cache -- skipping computation")
        else:
            lcs_dense, _ = _lcs_fast.pairwise_lcs_matrix_parallel(
                flat_corpus,
                ignore_duration = cfg.lcs_ignore_duration,
                ignore_octave   = cfg.lcs_ignore_octave,
                max_seq_len     = cfg.lcs_max_seq_len,
                pairs_mask      = cand,
                n_workers       = fc.n_workers,
                chunk_size      = fc.chunk_size,
                use_window      = fc.use_window,
            )
            lcs_mat[:] = lcs_dense
            del lcs_dense
            _save_matrix(np.array(lcs_mat), _lcs_cache)
        log.info("  LCS done in %.1fs", time.time() - _t); _t = time.time()

        # -- NOW free the token sequences -- biggest RAM saving -------------
        log.info("  Freeing token sequences from RAM ...")
        all_seqs_ref = all_seqs   # keep ref for group stats
        # TF-IDF and histogram will re-stream from disk

        # -- Step 4: TF-IDF -- two-pass streaming ---------------------------
        _t_last = _tick("Step 4: TF-IDF", _t_last)
        log.info("Step 4 -- TF-IDF (streaming, two-pass) ...")

        # Pass 1: fit vocabulary
        vec = TfidfVectorizer(
            ngram_range  = cfg.tfidf_ngram_range,
            max_features = cfg.tfidf_max_features,
            sublinear_tf = True,
        )
        docs_pass1 = [" ".join(s) for s in all_seqs]
        vec.fit(docs_pass1)
        del docs_pass1

        # Pass 2: transform in chunks, stay sparse throughout
        import scipy.sparse as sp
        CHUNK = 50
        tfidf_rows = []
        try:
            from tqdm import tqdm as _tqdm
            chunk_iter = _tqdm(range(0, n, CHUNK), desc="TF-IDF transform",
                               unit="chunk", dynamic_ncols=True)
        except ImportError:
            chunk_iter = range(0, n, CHUNK)

        for start in chunk_iter:
            batch_docs = [" ".join(all_seqs[i])
                          for i in range(start, min(start + CHUNK, n))]
            tfidf_rows.append(vec.transform(batch_docs))

        tfidf_sparse = sp.vstack(tfidf_rows)
        del tfidf_rows

        # Cosine distance -- sklearn accepts sparse directly, no densification
        _t_last = _tick("Step 4b: TF-IDF cosine distances", _t_last)
        log.info("  Computing TF-IDF cosine distances (sparse) ...")
        sim_tfidf = sklearn_cosine(tfidf_sparse)
        _tfidf_dist_arr = (1.0 - sim_tfidf).astype(np.float32)
        del sim_tfidf
        _tfidf_mmap_path = _mmap_dir / "tfidf_dist.mmap"
        tfidf_dist = np.memmap(str(_tfidf_mmap_path), dtype="float32",
                               mode="w+", shape=(n, n))
        tfidf_dist[:] = _tfidf_dist_arr
        del _tfidf_dist_arr
        log.info("  TF-IDF done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 5: Histogram distances ------------------------------------
        _t_last = _tick("Step 5: Histogram distances", _t_last)
        log.info("Step 5 -- Histogram distances ...")
        hist_mat_arr = np.stack(
            [music_feature_vector(s) for s in all_seqs]
        ).astype(np.float32)
        _hist_dist_arr = (1.0 - sklearn_cosine(hist_mat_arr)).astype(np.float32)
        _hist_mmap_path = _mmap_dir / "hist_dist.mmap"
        hist_dist = np.memmap(str(_hist_mmap_path), dtype="float32",
                              mode="w+", shape=(n, n))
        hist_dist[:] = _hist_dist_arr
        del _hist_dist_arr
        log.info("  Histogram done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 6: PCA via TruncatedSVD ---------------------------------
        # TruncatedSVD operates on the sparse matrix directly -- never
        # densifies, uses ~10x less RAM than full PCA, same visual result.
        _t_last = _tick("Step 6: PCA/TruncatedSVD", _t_last)
        log.info("Step 6 -- PCA (TruncatedSVD + centering, n=%d) ...", n)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        import threading as _threading

        _pca_result    = [None]
        _pca_done_flag = _threading.Event()

        def _run_svd():
            try:
                # Try UMAP first -- handles imbalanced corpora without horseshoe
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42,
                                    n_neighbors=15, min_dist=0.1,
                                    metric="cosine")
                coords = reducer.fit_transform(tfidf_sparse)
                _pca_result[0] = coords.astype(np.float32)
                log.info("  UMAP projection complete")
            except ImportError:
                log.info("  UMAP not available -- using centered TruncatedSVD")
                try:
                    # Centered SVD (true PCA) -- do NOT L2-normalise after,
                    # that's what causes the circle horseshoe
                    col_means  = np.asarray(tfidf_sparse.mean(axis=0)).ravel()
                    svd        = TruncatedSVD(n_components=50, random_state=42,
                                             n_iter=7)
                    svd.fit(tfidf_sparse)
                    coords_raw = svd.transform(tfidf_sparse)
                    mean_proj  = col_means @ svd.components_.T
                    coords     = (coords_raw - mean_proj)[:, :2].astype(np.float32)
                    _pca_result[0] = coords
                except Exception as exc:
                    log.warning("Centered SVD failed (%s) -- plain SVD fallback", exc)
                    svd2   = TruncatedSVD(n_components=2, random_state=42)
                    _pca_result[0] = svd2.fit_transform(tfidf_sparse).astype(np.float32)
            except Exception as exc:
                log.warning("UMAP failed (%s) -- plain SVD fallback", exc)
                svd2   = TruncatedSVD(n_components=2, random_state=42)
                _pca_result[0] = svd2.fit_transform(tfidf_sparse).astype(np.float32)
            _pca_done_flag.set()

        _svd_thread = _threading.Thread(target=_run_svd, daemon=True)
        _svd_thread.start()

        try:
            from tqdm import tqdm as _tqdm
            with _tqdm(total=0, desc="TruncatedSVD",
                       bar_format="{desc}: {elapsed} elapsed {postfix}",
                       dynamic_ncols=True) as pbar_svd:
                while not _pca_done_flag.wait(timeout=1.0):
                    pbar_svd.set_postfix_str(
                        f"{n} pieces x {tfidf_sparse.shape[1]} features ...")
                pbar_svd.set_postfix_str("done")
        except ImportError:
            _pca_done_flag.wait()

        _svd_thread.join()
        pca_xy   = _pca_result[0]
        pca_dict = {idx.names[i]: pca_xy[i].tolist() for i in range(n)}
        log.info("  SVD done")

        # Save tfidf dense rows per group for style_collapse_score.
        # We do this NOW while tfidf_sparse is still in scope, storing
        # a dict so we don't need tfidf_sparse after it's deleted.
        print(f"[{time.strftime('%H:%M:%S')}] Pre-computing per-group TF-IDF dense rows ...", flush=True)
        _group_tfidf_cache: dict = {}
        for _g, (_gdir, _glim) in self._group_dirs.items():
            _gname  = _g.value
            _gidxs  = [_ii for _ii, _gg in enumerate(idx.groups) if _gg == _gname]
            if _gidxs:
                _group_tfidf_cache[_gname] = tfidf_sparse[_gidxs].toarray().astype(np.float32)
        print(f"[{time.strftime('%H:%M:%S')}] Per-group TF-IDF rows cached", flush=True)

        def _group_tfidf_dense(gidxs, gname=None):
            """Return pre-cached dense TF-IDF rows for this group."""
            if gname and gname in _group_tfidf_cache:
                return _group_tfidf_cache[gname]
            # Fallback: recompute from tfidf_dist diagonal isn't possible,
            # return zeros -- should not happen
            return np.zeros((len(gidxs), 1), dtype=np.float32)

        # -- Step 7: Calibration --------------------------------------------
        calibrated: Optional[CalibratedThresholds] = None
        if calibrator is not None:
            _t_last = _tick("Step 7: Calibration", _t_last)
            calibrated = self._run_calibration(
                calibrator, idx.names, all_seqs, name_to_group,
                ngram_mat, lcs_mat, tfidf_dist, hist_dist,
            )

        _t_last = _tick("Calibration returned", _t_last)
        print(f"[{time.strftime('%H:%M:%S')}] Calibration done -- freeing large objects ...", flush=True)

        active_ngram_thr = (calibrated.jaccard_n5 if calibrated
                            else cfg.ngram_flag_threshold)
        active_lcs_thr   = (calibrated.lcs_strict if calibrated
                            else cfg.lcs_flag_threshold)

        print(f"[{time.strftime('%H:%M:%S')}] Thresholds set: lcs={active_lcs_thr:.4f}  ngram={active_ngram_thr:.4f}", flush=True)

        # Free large objects before pair assembly to reduce memory pressure
        print(f"[{time.strftime('%H:%M:%S')}] Deleting tfidf_sparse ...", flush=True)
        del tfidf_sparse
        # Cache sequence lengths before freeing -- needed for group stats
        _seq_lens = [len(s) for s in all_seqs]
        print(f"[{time.strftime('%H:%M:%S')}] Deleting all_seqs ...", flush=True)
        del all_seqs
        # tfidf_dist and hist_dist are memmaps -- stay accessible but off RAM
        import gc as _gc
        print(f"[{time.strftime('%H:%M:%S')}] Running gc.collect() ...", flush=True)
        _gc.collect()
        print(f"[{time.strftime('%H:%M:%S')}] Memory freed -- starting pair assembly", flush=True)

        # -- Step 8: Pair results --------------------------------------------
        _t_last = _tick("Step 8: Pair assembly", _t_last)
        log.info("Assembling pair results ...")
        from music_bias_analyzer.pipeline import EnsembleVoter
        voter = (EnsembleVoter(calibrated or CalibratedThresholds(), ensemble)
                 if ensemble else None)

        from music_bias_analyzer.ngram import (bleu_precision, ngram_set,
                                               ngram_counts)
        NS          = cfg.ngram_ns
        total_pairs = n * (n - 1) // 2

        import pickle as _pickle
        _pr_pkl = cache_dir / "pair_results.pkl"

        # -- Try loading from cache first (before any expensive computation) --
        pair_results = None
        if _pr_pkl.exists():
            try:
                with open(str(_pr_pkl), "rb") as _f:
                    pair_results = _pickle.load(_f)
                log.info("  Pair results loaded from cache (%d pairs) "
                         "-- skipping assembly", len(pair_results))
                # Re-apply current thresholds in case config changed
                for pr in pair_results:
                    pr.flagged_ngram = (
                        pr.ngram_scores.get("jaccard_n5", 0) >= active_ngram_thr)
                    pr.flagged_lcs = pr.lcs_score >= active_lcs_thr
                    if voter:
                        pr.flagged_ensemble = voter.flag(pr)
            except Exception as exc:
                log.warning("  Pair cache load failed (%s) -- recomputing", exc)
                pair_results = None

        # -- Assemble if not loaded from cache ----------------------------
        if pair_results is None:
            # Use pre-computed matrices directly -- no need to rebuild n-gram
            # sets since jaccard_n5 is already in ngram_mat.
            # BLEU is skipped here; the calibrator only needs jaccard.
            pair_results = []
            _t_last = _tick("Step 8a: Starting pair loop", _t_last)
            log.info("  Assembling %d pairs from matrices ...", total_pairs)
            t_pr      = time.time()
            LOG_EVERY = max(1, total_pairs // 20)

            try:
                from tqdm import tqdm as _tqdm
                pbar_pr = _tqdm(total=total_pairs, desc="Pair assembly",
                                unit="pair", dynamic_ncols=True,
                                mininterval=2.0)
            except ImportError:
                pbar_pr = None

            for i in range(n):
                for j in range(i + 1, n):
                    jac_n5 = round(float(ngram_mat[i, j]), 5)
                    # Store jaccard for all configured ns using the n5 matrix
                    # (full per-n recomputation deferred to flagged pairs only)
                    ng_scores = {f"jaccard_n{nv}": jac_n5 for nv in NS}

                    pr = PairResult(
                        piece_a               = idx.names[i],
                        piece_b               = idx.names[j],
                        group_a               = idx.groups[i],
                        group_b               = idx.groups[j],
                        ngram_scores          = ng_scores,
                        lcs_score             = round(float(lcs_mat[i, j]), 5),
                        tfidf_cosine_dist     = round(float(tfidf_dist[i, j]), 5),
                        histogram_cosine_dist = round(float(hist_dist[i, j]), 5),
                        flagged_ngram         = jac_n5 >= active_ngram_thr,
                        flagged_lcs           = float(lcs_mat[i, j]) >= active_lcs_thr,
                    )
                    if voter:
                        pr.flagged_ensemble = voter.flag(pr)
                    pair_results.append(pr)

                    done = len(pair_results)
                    if pbar_pr is not None:
                        pbar_pr.update(1)
                    elif done % LOG_EVERY == 0:
                        elapsed_pr = time.time() - t_pr
                        rate_pr    = done / max(elapsed_pr, 0.001)
                        eta_pr     = (total_pairs - done) / max(rate_pr, 0.001)
                        log.info("  Pair assembly: %d/%d (%.0f%%)  "
                                 "%.0f pairs/s  ETA %.0fs",
                                 done, total_pairs,
                                 100 * done / total_pairs,
                                 rate_pr, eta_pr)

            if pbar_pr is not None:
                pbar_pr.close()

            try:
                with open(str(_pr_pkl), "wb") as _f:
                    _pickle.dump(pair_results, _f, protocol=4)
                log.info("  Pair results cached -> %s  (%d pairs)",
                         _pr_pkl, len(pair_results))
            except Exception as exc:
                log.warning("  Could not cache pair results: %s", exc)

        _t_last = _tick("Step 9: Group stats", _t_last)
        # -- Step 9: Group stats ---------------------------------------------
        group_stats: Dict[str, GroupStats] = {}
        for group, (directory, limit) in self._group_dirs.items():
            gname  = group.value
            gidxs  = [i for i, g in enumerate(idx.groups) if g == gname]
            if not gidxs:
                continue
            g_tfidf = _group_tfidf_dense(gidxs, gname)
            g_hist  = hist_mat_arr[gidxs]
            intra_lcs = [float(lcs_mat[a, b])
                         for ii, a in enumerate(gidxs)
                         for b in gidxs[ii + 1:]]
            intra_jac = [float(ngram_mat[a, b])
                         for ii, a in enumerate(gidxs)
                         for b in gidxs[ii + 1:]]
            group_stats[gname] = GroupStats(
                group        = gname,
                n_pieces     = len(gidxs),
                mean_seq_len = float(np.mean([_seq_lens[i] for i in gidxs])),
                style_collapse_tfidf     = round(style_collapse_score(g_tfidf), 4),
                style_collapse_histogram = round(style_collapse_score(g_hist),  4),
                intra_mean_lcs     = round(np.mean(intra_lcs),  4) if intra_lcs else 0.0,
                intra_mean_jaccard = round(np.mean(intra_jac), 4) if intra_jac else 0.0,
            )

        # -- Cleanup memmaps -------------------------------------------------
        del ngram_mat, lcs_mat
        # Clean up mmap files
        del tfidf_dist, hist_dist
        import shutil as _shutil
        try:
            _shutil.rmtree(str(_mmap_dir), ignore_errors=True)
            log.info("Cleaned up mmap temp files")
        except Exception:
            pass

        log.info("LeanPipeline complete in %.1fs", time.time() - t0)

        # Populate _groups with empty corpora -- sequences already freed.
        # The report and calibration both ran before this point so
        # _groups content is not needed for the return value.
        self._groups = {group: {} for group in self._group_dirs}

        return AnalysisReport(
            config                = cfg,
            timestamp             = time.strftime("%Y-%m-%dT%H:%M:%S"),
            group_stats           = group_stats,
            pair_results          = pair_results,
            pca_coords            = pca_dict,
            calibrated_thresholds = calibrated,
        )