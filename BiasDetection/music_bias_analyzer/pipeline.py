"""
pipeline.py
-----------
Orchestrates the full symbolic-music bias analysis pipeline.

Experiment groups (mirrors the NotaGen bias study design):
  - LLM_OUTPUT    : generated pieces under evaluation (e.g. NotaGen outputs)
  - TRAIN_CORPUS  : fine-tuning / pre-training data (suspected source)
  - DUPED         : manually documented suspected duplicates
  - HUMAN_BASELINE: held-out human-composed reference (e.g. MAESTRO subset)

The pipeline runs all three metric families and aggregates results into a
single AnalysisReport that can be saved to JSON or rendered as a summary.

Two threshold strategies are supported:

  Manual (default)
    Fixed thresholds in PipelineConfig.  Sensible starting points based on
    the literature, but not calibrated to your specific corpus.

  Pareto calibration  [R8 -- Malandrino et al. 2022]
    ThresholdCalibrator sweeps all observed score values as candidate
    thresholds, computing TP/FP rates against the human baseline null
    distribution, and selects Pareto-optimal thresholds that minimize
    false positives while maximizing true positives.  Call
    pipeline.run(calibrator=ThresholdCalibrator()) to activate.

Ensemble voting  [R9 -- Malandrino et al. 2022]
    EnsembleVoter applies AND logic across multiple metrics: a pair is
    flagged only when it exceeds threshold on ALL configured metrics.
    Reduces false positive rate compared to single-metric flagging.
    Call pipeline.run(ensemble=EnsembleConfig(...)) to activate.

Usage
-----
    from music_bias_analyzer.pipeline import (
        AnalysisPipeline, CorpusGroup,
        ThresholdCalibrator, EnsembleConfig,
    )

    pipeline = AnalysisPipeline()
    pipeline.add_corpus(CorpusGroup.LLM_OUTPUT,     "path/to/notagen_outputs/")
    pipeline.add_corpus(CorpusGroup.TRAIN_CORPUS,   "path/to/train_corpus/")
    pipeline.add_corpus(CorpusGroup.HUMAN_BASELINE, "path/to/maestro_subset/")

    # Basic run (manual thresholds)
    report = pipeline.run()

    # With Pareto calibration + ensemble voting
    calibrator = ThresholdCalibrator()
    ensemble   = EnsembleConfig(metrics=["jaccard_n5", "lcs_strict"])
    report     = pipeline.run(calibrator=calibrator, ensemble=ensemble)

    report.save("results/report.json")
    print(report.summary())
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tokenizer  import load_corpus, file_to_tokens
from .ngram      import (pairwise_ngram_matrix, flag_high_overlap_pairs,
                         multi_n_overlap, candidate_pairs)
from .embeddings import (TFIDFMusicEmbedder, corpus_feature_matrix,
                         cosine_distance_matrix, style_collapse_score, pca_2d)
from .lcs        import pairwise_lcs_matrix, lcs_density_score

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus group taxonomy
# ---------------------------------------------------------------------------

class CorpusGroup(str, Enum):
    LLM_OUTPUT      = "llm_output"
    TRAIN_CORPUS    = "train_corpus"
    DUPED           = "duped"
    HUMAN_BASELINE  = "human_baseline"
    OTHER           = "other"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # N-gram settings
    ngram_ns: List[int]        = field(default_factory=lambda: [3, 5, 8])
    ngram_flag_threshold: float = 0.15   # Jaccard @ n=5 to flag a pair

    # LCS settings
    lcs_ignore_duration: bool  = False
    lcs_ignore_octave: bool    = False
    lcs_max_seq_len: int       = 3000
    lcs_flag_threshold: float  = 0.35
    lcs_cross_group_only: bool = False  # skip intra-group LCS (big speedup for large baselines)

    # Embedding settings
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_max_features: int            = 5000

    # Token settings
    use_octave:   bool = True
    use_duration: bool = True


# ---------------------------------------------------------------------------
# Calibrated thresholds  [R8 -- Malandrino et al. 2022]
# ---------------------------------------------------------------------------

@dataclass
class CalibratedThresholds:
    """
    Per-metric thresholds, either manually set or Pareto-calibrated.

    For HIGH-similarity metrics (jaccard, lcs): flag if score ABOVE threshold.
    For LOW-distance metrics (tfidf_dist, hist_dist): flag if score BELOW threshold.

    calibration_source : "manual" | "pareto"
    n_null_pairs       : number of human-baseline pairs used for calibration
    n_candidate_pairs  : number of llmxtrain pairs used as positive candidates
    """
    # High-similarity thresholds (flag if score > value)
    jaccard_n5:   float = 0.15
    bleu_n5:      float = 0.20
    lcs_strict:   float = 0.35
    lcs_pitch:    float = 0.40
    # Low-distance thresholds (flag if score < value)
    tfidf_dist_max: float = 0.25
    hist_dist_max:  float = 0.10
    # Provenance
    calibration_source:  str = "manual"
    n_null_pairs:        int = 0
    n_candidate_pairs:   int = 0


class ThresholdCalibrator:
    """
    Pareto-front threshold calibration.

    Source: Malandrino et al. (2022) -- adaptive meta-heuristic threshold
    selection on the Pareto frontier of TP vs FP rate.

    Concept
    -------
    For each metric we have two score distributions:
      - NULL  : human-baseline pairs (the "not suspicious" reference)
      - CAND  : llmxtrain pairs (the candidates we want to flag)

    We sweep every observed score value as a candidate threshold T.  For
    each T we compute:
      - FP_rate = fraction of NULL pairs exceeding T  (false positives)
      - TP_rate = fraction of CAND pairs exceeding T  (true positives)

    The Pareto front contains all thresholds where no other T achieves
    both higher TP_rate AND lower FP_rate simultaneously.  We select the
    Pareto-optimal T that minimizes FP_rate (most conservative), which is
    appropriate for a scientific bias audit where false positives are costly.

    For LOW-distance metrics (tfidf_dist, hist_dist), the logic is inverted:
    we flag pairs BELOW the threshold (small distance = suspicious similarity).

    Usage
    -----
        calibrator = ThresholdCalibrator()
        thresholds = calibrator.calibrate(null_scores, candidate_scores)

    Where null_scores and candidate_scores are dicts of:
        { "jaccard_n5": [0.02, 0.05, ...], "lcs_strict": [0.12, ...], ... }

    The pipeline.run() method extracts these automatically from the loaded
    corpus groups when a calibrator is provided.
    """

    # Metrics where HIGH score = suspicious
    HIGH_METRICS = ("jaccard_n5", "bleu_n5", "lcs_strict", "lcs_pitch")
    # Metrics where LOW distance = suspicious (inverted sweep)
    LOW_METRICS  = ("tfidf_dist", "hist_dist")

    def calibrate(self,
                  null_scores:      Dict[str, List[float]],
                  candidate_scores: Dict[str, List[float]]
                  ) -> CalibratedThresholds:
        """
        Run Pareto sweep and return calibrated thresholds.

        Parameters
        ----------
        null_scores      : {metric: [scores]} for human-baseline pairs
        candidate_scores : {metric: [scores]} for llmxtrain pairs

        Returns
        -------
        CalibratedThresholds with calibration_source="pareto"
        """
        result = CalibratedThresholds(
            calibration_source = "pareto",
            n_null_pairs       = len(next(iter(null_scores.values()), [])),
            n_candidate_pairs  = len(next(iter(candidate_scores.values()), [])),
        )

        for metric in self.HIGH_METRICS:
            null_vals = null_scores.get(metric, [])
            cand_vals = candidate_scores.get(metric, [])
            if not null_vals or not cand_vals:
                log.warning("ThresholdCalibrator: no data for metric '%s', "
                            "keeping manual default", metric)
                continue
            t = self._pareto_threshold_high(null_vals, cand_vals)
            setattr(result, metric, round(t, 4))
            log.info("Calibrated %s threshold: %.4f", metric, t)

        for metric in self.LOW_METRICS:
            null_vals = null_scores.get(metric, [])
            cand_vals = candidate_scores.get(metric, [])
            if not null_vals or not cand_vals:
                log.warning("ThresholdCalibrator: no data for metric '%s', "
                            "keeping manual default", metric)
                continue
            t = self._pareto_threshold_low(null_vals, cand_vals)
            attr = f"{metric}_max"
            setattr(result, attr, round(t, 4))
            log.info("Calibrated %s_max threshold: %.4f", metric, t)

        return result

    @staticmethod
    def _pareto_threshold_high(null_vals: List[float],
                                cand_vals: List[float]) -> float:
        """
        For HIGH-similarity metrics: find the Pareto-optimal threshold
        that minimizes FP_rate (fraction of null pairs above T) while
        maximizing TP_rate (fraction of candidate pairs above T).

        Returns the most conservative Pareto-optimal threshold (lowest FP).
        Falls back to mean(null) + 2*std(null) if the Pareto front is empty.
        """
        null_arr = np.array(null_vals, dtype=float)
        cand_arr = np.array(cand_vals, dtype=float)

        # All unique score values across both distributions as candidates
        candidates = sorted(set(null_vals) | set(cand_vals))

        pareto: List[Tuple[float, float, float]] = []  # (T, fp, tp)
        for T in candidates:
            fp = float(np.mean(null_arr >= T))
            tp = float(np.mean(cand_arr >= T))
            pareto.append((T, fp, tp))

        # Pareto front: keep (T, fp, tp) where no other point has
        # strictly lower fp AND strictly higher tp
        front = []
        for T, fp, tp in pareto:
            dominated = any(
                fp2 <= fp and tp2 >= tp and (fp2 < fp or tp2 > tp)
                for _, fp2, tp2 in pareto
            )
            if not dominated:
                front.append((T, fp, tp))

        if not front:
            # Fallback: null mean + 2 std
            return float(null_arr.mean() + 2 * null_arr.std())

        # Most conservative = lowest FP rate among Pareto-optimal points
        best_T, best_fp, best_tp = min(front, key=lambda x: x[1])
        return float(best_T)

    @staticmethod
    def _pareto_threshold_low(null_vals: List[float],
                               cand_vals: List[float]) -> float:
        """
        For LOW-distance metrics (smaller = more suspicious):
        find the threshold T below which candidate pairs cluster
        but null pairs do not.  Inverts the HIGH logic.
        """
        null_arr = np.array(null_vals, dtype=float)
        cand_arr = np.array(cand_vals, dtype=float)

        candidates = sorted(set(null_vals) | set(cand_vals))

        pareto: List[Tuple[float, float, float]] = []
        for T in candidates:
            fp = float(np.mean(null_arr <= T))   # null pairs wrongly flagged
            tp = float(np.mean(cand_arr <= T))   # candidate pairs correctly flagged
            pareto.append((T, fp, tp))

        front = []
        for T, fp, tp in pareto:
            dominated = any(
                fp2 <= fp and tp2 >= tp and (fp2 < fp or tp2 > tp)
                for _, fp2, tp2 in pareto
            )
            if not dominated:
                front.append((T, fp, tp))

        if not front:
            return float(null_arr.mean() - 2 * null_arr.std())

        best_T, _, _ = min(front, key=lambda x: x[1])
        return float(best_T)


# ---------------------------------------------------------------------------
# Ensemble voting  [R9 -- Malandrino et al. 2022]
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble metric voting.

    metrics : list of metric names to include in the AND gate.
              Valid names: "jaccard_n5", "bleu_n5", "lcs_strict",
              "lcs_pitch", "tfidf_dist", "hist_dist".
              Default: all six metrics.
    require_all : True = AND logic (all must flag); False = OR logic (any flags).
                  AND logic is recommended -- reduces false positives.

    Source: Malandrino et al. (2022) ensemble of cosine + Dice + Overlap
    metrics with Pareto-calibrated thresholds achieving 88% accuracy.
    """
    metrics: List[str] = field(default_factory=lambda: [
        "jaccard_n5", "lcs_strict", "tfidf_dist"
    ])
    require_all: bool = True   # AND logic


class EnsembleVoter:
    """
    Applies ensemble AND/OR flagging logic to a PairResult.

    Instantiate with calibrated thresholds and an EnsembleConfig,
    then call flag(pair_result) on each pair.
    """

    def __init__(self, thresholds: CalibratedThresholds,
                 config: EnsembleConfig):
        self.thresholds = thresholds
        self.config     = config

    def flag(self, pair: "PairResult") -> bool:
        """
        Return True if the pair should be flagged under ensemble logic.

        For each metric in config.metrics:
          - HIGH metrics: flag if score > threshold
          - LOW  metrics: flag if score < threshold

        If require_all=True  (AND): ALL metrics must flag -> True
        If require_all=False (OR):  ANY metric  flags  -> True
        """
        votes: List[bool] = []
        thr = self.thresholds

        for metric in self.config.metrics:
            if metric == "jaccard_n5":
                score = pair.ngram_scores.get("jaccard_n5", 0.0)
                votes.append(score > thr.jaccard_n5)
            elif metric == "bleu_n5":
                score = pair.ngram_scores.get("bleu_n5", 0.0)
                votes.append(score > thr.bleu_n5)
            elif metric == "lcs_strict":
                votes.append(pair.lcs_score > thr.lcs_strict)
            elif metric == "lcs_pitch":
                votes.append(pair.lcs_score > thr.lcs_pitch)
            elif metric == "tfidf_dist":
                if pair.tfidf_cosine_dist is not None:
                    votes.append(pair.tfidf_cosine_dist < thr.tfidf_dist_max)
            elif metric == "hist_dist":
                if pair.histogram_cosine_dist is not None:
                    votes.append(pair.histogram_cosine_dist < thr.hist_dist_max)
            else:
                log.warning("EnsembleVoter: unknown metric '%s', skipping", metric)

        if not votes:
            return False
        return all(votes) if self.config.require_all else any(votes)

@dataclass
class PairResult:
    piece_a:      str
    piece_b:      str
    group_a:      str
    group_b:      str
    ngram_scores: Dict[str, float]   # keys: "jaccard_n3", "bleu_n5", ...
    lcs_score:    float
    tfidf_cosine_dist: Optional[float]
    histogram_cosine_dist: Optional[float]
    flagged_ngram:    bool
    flagged_lcs:      bool
    flagged_ensemble: bool = False   # [R9] set by EnsembleVoter after run()

    @property
    def is_flagged(self) -> bool:
        return self.flagged_ngram or self.flagged_lcs

    @property
    def is_flagged_ensemble(self) -> bool:
        return self.flagged_ensemble


@dataclass
class GroupStats:
    group:              str
    n_pieces:           int
    mean_seq_len:       float
    style_collapse_tfidf:     float   # lower = more collapsed
    style_collapse_histogram: float
    intra_mean_lcs:     float         # mean LCS within group
    intra_mean_jaccard: float         # mean Jaccard (n=5) within group


@dataclass
class AnalysisReport:
    config:       PipelineConfig
    timestamp:    str
    group_stats:  Dict[str, GroupStats]
    pair_results: List[PairResult]
    pca_coords:   Optional[Dict[str, List[float]]]   # name -> [x, y]
    flagged_pairs:          List[PairResult] = field(default_factory=list)
    ensemble_flagged_pairs: List[PairResult] = field(default_factory=list)   # [R9]
    calibrated_thresholds:  Optional[CalibratedThresholds] = None            # [R8]

    def __post_init__(self):
        self.flagged_pairs          = [p for p in self.pair_results if p.is_flagged]
        self.ensemble_flagged_pairs = [p for p in self.pair_results if p.is_flagged_ensemble]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MUSIC BIAS ANALYSIS REPORT",
            f"Timestamp : {self.timestamp}",
            f"Total pairs analyzed    : {len(self.pair_results)}",
            f"Flagged pairs (any)     : {len(self.flagged_pairs)}",
            f"Flagged pairs (ensemble): {len(self.ensemble_flagged_pairs)}",
        ]
        if self.calibrated_thresholds:
            ct = self.calibrated_thresholds
            lines.append(
                f"Threshold calibration   : {ct.calibration_source}  "
                f"(null_pairs={ct.n_null_pairs}, "
                f"candidate_pairs={ct.n_candidate_pairs})"
            )
        lines += [
            "",
            "-- Group Statistics --",
        ]
        for gs in self.group_stats.values():
            lines += [
                f"  [{gs.group}]  n={gs.n_pieces}  "
                f"mean_len={gs.mean_seq_len:.0f}  "
                f"style_collapse(tfidf)={gs.style_collapse_tfidf:.3f}  "
                f"intra_lcs={gs.intra_mean_lcs:.3f}",
            ]
        lines += ["", "-- Top 10 Flagged Pairs --"]
        for p in sorted(self.flagged_pairs,
                        key=lambda x: x.lcs_score, reverse=True)[:10]:
            lines.append(
                f"  {p.piece_a}  <->  {p.piece_b}  "
                f"[{p.group_a}->{p.group_b}]  "
                f"jaccard_n5={p.ngram_scores.get('jaccard_n5',0):.3f}  "
                f"lcs={p.lcs_score:.3f}"
            )
        if self.ensemble_flagged_pairs:
            lines += ["", "-- Ensemble-Flagged Pairs (AND logic) --"]
            for p in sorted(self.ensemble_flagged_pairs,
                            key=lambda x: x.lcs_score, reverse=True)[:10]:
                lines.append(
                    f"  * {p.piece_a}  <->  {p.piece_b}  "
                    f"[{p.group_a}->{p.group_b}]  "
                    f"jaccard_n5={p.ngram_scores.get('jaccard_n5',0):.3f}  "
                    f"lcs={p.lcs_score:.3f}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp":              self.timestamp,
            "calibrated_thresholds":  asdict(self.calibrated_thresholds)
                                      if self.calibrated_thresholds else None,
            "group_stats":            {k: asdict(v) for k, v in self.group_stats.items()},
            "pair_results":           [asdict(p) for p in self.pair_results],
            "flagged_pairs":          [asdict(p) for p in self.flagged_pairs],
            "ensemble_flagged_pairs": [asdict(p) for p in self.ensemble_flagged_pairs],
            "pca_coords":             self.pca_coords,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Report saved to %s", path)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class AnalysisPipeline:
    """
    End-to-end bias analysis pipeline.

    Step 1: Load corpora -> token sequences
    Step 2: N-gram pairwise matrix (verbatim copy detection)
    Step 3: LCS pairwise matrix    (paraphrase detection)
    Step 4: TF-IDF + histogram embeddings (style collapse)
    Step 5: Aggregate results into AnalysisReport
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config  = config or PipelineConfig()
        self._groups: Dict[CorpusGroup, Dict[str, List[str]]] = {}

    # -- Corpus loading ----------------------------------------------------

    def add_corpus(self, group: CorpusGroup, directory: str | Path,
                   limit: int = 0) -> "AnalysisPipeline":
        """Load all music files from a directory into the given group.

        limit : stop after this many successfully parsed files (0 = no limit).
                Passed directly to load_corpus() so files beyond the limit
                are never read from disk.
        """
        corpus = load_corpus(
            directory,
            limit        = limit,
            use_octave   = self.config.use_octave,
            use_duration = self.config.use_duration,
        )
        if not corpus:
            log.warning("No parseable files found in %s", directory)
        self._groups[group] = corpus
        log.info("Loaded %d pieces into group '%s'", len(corpus), group.value)
        return self

    def add_file(self, group: CorpusGroup, filepath: str | Path,
                 name: Optional[str] = None) -> "AnalysisPipeline":
        """Add a single file to a group."""
        fp   = Path(filepath)
        name = name or fp.name
        toks = file_to_tokens(fp, use_octave=self.config.use_octave,
                              use_duration=self.config.use_duration)
        if not toks:
            log.warning("Could not tokenize %s", fp)
        else:
            self._groups.setdefault(group, {})[name] = toks
        return self

    def add_tokens(self, group: CorpusGroup, name: str,
                   tokens: List[str]) -> "AnalysisPipeline":
        """Add pre-tokenized sequence directly (useful for LLM output piped in-memory)."""
        self._groups.setdefault(group, {})[name] = tokens
        return self

    # -- Run --------------------------------------------------------------

    def run(self,
            calibrator: Optional[ThresholdCalibrator] = None,
            ensemble:   Optional[EnsembleConfig]      = None,
            ) -> AnalysisReport:
        """
        Execute the full analysis pipeline.

        Parameters
        ----------
        calibrator : ThresholdCalibrator, optional
            When provided, derives Pareto-optimal thresholds from the
            human-baseline null distribution before flagging pairs.  [R8]
        ensemble : EnsembleConfig, optional
            When provided, applies AND/OR voting across multiple metrics and
            sets flagged_ensemble on each PairResult.  Requires calibrator
            or falls back to PipelineConfig manual thresholds.  [R9]
        """
        if not self._groups:
            raise ValueError("No corpora loaded. Call add_corpus() first.")

        t0 = time.time()
        log.info("Pipeline starting ...")

        # Flatten all pieces with group labels
        all_names:  List[str] = []
        all_seqs:   List[List[str]] = []
        name_to_group: Dict[str, str] = {}

        for group, corpus in self._groups.items():
            for name, seq in corpus.items():
                all_names.append(name)
                all_seqs.append(seq)
                name_to_group[name] = group.value

        n = len(all_names)
        log.info("Total pieces: %d  (step 0 done in %.1fs)", n, time.time() - t0)
        _t = time.time()

        # -- Step 1: N-gram matrix --
        log.info("Step 1/5 -- N-gram matrices ...")
        flat_corpus = dict(zip(all_names, all_seqs))
        ngram_mat, _ = pairwise_ngram_matrix(
            flat_corpus, n=5, metric="jaccard")
        log.info("Step 1 done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 2: LCS matrix (n-gram pre-filtered) --
        log.info("Computing LCS matrix ...")
        lcs_candidates = set(
            candidate_pairs(flat_corpus, min_jaccard=0.03, n=3)
        )

        # Build pairs_mask if cross_group_only is requested
        lcs_pairs_mask = None
        if self.config.lcs_cross_group_only:
            baseline_grp = CorpusGroup.HUMAN_BASELINE.value
            idx_map      = {name: i for i, name in enumerate(all_names)}
            lcs_pairs_mask = set()
            for i, na in enumerate(all_names):
                for j, nb in enumerate(all_names):
                    if j <= i:
                        continue
                    ga, gb = name_to_group[na], name_to_group[nb]
                    # Skip pairs where BOTH pieces are human baseline
                    if ga == baseline_grp and gb == baseline_grp:
                        continue
                    lcs_pairs_mask.add((i, j))
            log.info(
                "cross_group_only: LCS will compute %d pairs (skipping humanxhuman)",
                len(lcs_pairs_mask),
            )

        lcs_mat, _ = pairwise_lcs_matrix(
            flat_corpus,
            ignore_duration = self.config.lcs_ignore_duration,
            ignore_octave   = self.config.lcs_ignore_octave,
            max_seq_len     = self.config.lcs_max_seq_len,
            pairs_mask      = lcs_pairs_mask,
        )
        log.info("Step 2 done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 3: Embeddings --
        log.info("Step 3/5 -- TF-IDF embeddings ...")
        embedder = TFIDFMusicEmbedder(
            ngram_range  = self.config.tfidf_ngram_range,
            max_features = self.config.tfidf_max_features,
        )
        tfidf_mat, emb_names = embedder.fit_transform(flat_corpus)
        tfidf_dist = cosine_distance_matrix(tfidf_mat)
        log.info("Step 3a (TF-IDF) done in %.1fs", time.time() - _t); _t = time.time()

        log.info("Step 3b -- pitch-class histogram embeddings ...")
        hist_mat, _ = corpus_feature_matrix(flat_corpus)
        hist_dist   = cosine_distance_matrix(hist_mat)
        log.info("Step 3b done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 4: PCA coords for visualization --
        log.info("Step 4/5 -- PCA ...")
        pca_xy   = pca_2d(tfidf_mat)
        pca_dict = {name: pca_xy[i].tolist()
                    for i, name in enumerate(all_names)}
        log.info("Step 4 done in %.1fs", time.time() - _t); _t = time.time()

        # -- Step 5: Pareto threshold calibration [R8] --
        log.info("Step 5/5 -- calibration + pair assembly ...")
        calibrated: Optional[CalibratedThresholds] = None
        if calibrator is not None:
            log.info("Running Pareto threshold calibration ...")
            calibrated = self._run_calibration(
                calibrator, all_names, all_seqs, name_to_group,
                ngram_mat, lcs_mat, tfidf_dist, hist_dist,
            )

        # Resolve active thresholds (calibrated overrides manual)
        active_ngram_thr = (calibrated.jaccard_n5
                            if calibrated else self.config.ngram_flag_threshold)
        active_lcs_thr   = (calibrated.lcs_strict
                            if calibrated else self.config.lcs_flag_threshold)

        # -- Step 6: Build per-pair results --
        log.info("Assembling pair results ...")
        pair_results: List[PairResult] = []
        idx = {name: i for i, name in enumerate(all_names)}

        for i in range(n):
            for j in range(i + 1, n):
                na, nb = all_names[i], all_names[j]
                ia, ib = idx[na], idx[nb]

                j5   = float(ngram_mat[ia, ib])
                lcs  = float(lcs_mat[ia, ib])
                td   = float(tfidf_dist[ia, ib])
                hd   = float(hist_dist[ia, ib])

                ng_scores = multi_n_overlap(all_seqs[ia], all_seqs[ib],
                                           ns=self.config.ngram_ns)

                pair_results.append(PairResult(
                    piece_a   = na,
                    piece_b   = nb,
                    group_a   = name_to_group[na],
                    group_b   = name_to_group[nb],
                    ngram_scores          = {k: round(v, 5) for k, v in ng_scores.items()},
                    lcs_score             = round(lcs, 5),
                    tfidf_cosine_dist     = round(td, 5),
                    histogram_cosine_dist = round(hd, 5),
                    flagged_ngram = j5  >= active_ngram_thr,
                    flagged_lcs   = lcs >= active_lcs_thr,
                ))

        # -- Step 7: Ensemble voting [R9] --
        if ensemble is not None:
            thr    = calibrated or CalibratedThresholds()
            voter  = EnsembleVoter(thresholds=thr, config=ensemble)
            log.info("Applying ensemble voting (%s, require_all=%s) ...",
                     ensemble.metrics, ensemble.require_all)
            for pair in pair_results:
                pair.flagged_ensemble = voter.flag(pair)

        # -- Step 8: Group-level stats --
        group_stats: Dict[str, GroupStats] = {}
        for group, corpus in self._groups.items():
            gnames = list(corpus.keys())
            gidxs  = [idx[nm] for nm in gnames if nm in idx]
            if not gidxs:
                continue

            g_tfidf = tfidf_mat[gidxs]
            g_hist  = hist_mat[gidxs]

            intra_lcs = []
            intra_jac = []
            for ii in range(len(gidxs)):
                for jj in range(ii + 1, len(gidxs)):
                    intra_lcs.append(float(lcs_mat[gidxs[ii], gidxs[jj]]))
                    intra_jac.append(float(ngram_mat[gidxs[ii], gidxs[jj]]))

            group_stats[group.value] = GroupStats(
                group        = group.value,
                n_pieces     = len(gidxs),
                mean_seq_len = float(np.mean([len(corpus[nm]) for nm in gnames])),
                style_collapse_tfidf     = round(style_collapse_score(g_tfidf), 4),
                style_collapse_histogram = round(style_collapse_score(g_hist),  4),
                intra_mean_lcs     = round(np.mean(intra_lcs),  4) if intra_lcs else 0.0,
                intra_mean_jaccard = round(np.mean(intra_jac), 4) if intra_jac else 0.0,
            )

        elapsed = time.time() - t0
        log.info("Pipeline complete in %.1fs", elapsed)

        return AnalysisReport(
            config                = self.config,
            timestamp             = time.strftime("%Y-%m-%dT%H:%M:%S"),
            group_stats           = group_stats,
            pair_results          = pair_results,
            pca_coords            = pca_dict,
            calibrated_thresholds = calibrated,
        )

    def _run_calibration(self,
                         calibrator:     ThresholdCalibrator,
                         all_names:      List[str],
                         all_seqs:       List[List[str]],
                         name_to_group:  Dict[str, str],
                         ngram_mat:      np.ndarray,
                         lcs_mat:        np.ndarray,
                         tfidf_dist:     np.ndarray,
                         hist_dist:      np.ndarray,
                         ) -> CalibratedThresholds:
        """
        Extract null (humanxhuman) and candidate (llmxtrain) score
        distributions from the precomputed matrices, then calibrate.
        """
        idx = {name: i for i, name in enumerate(all_names)}

        null_scores:      Dict[str, List[float]] = {m: [] for m in
                          ThresholdCalibrator.HIGH_METRICS + ThresholdCalibrator.LOW_METRICS}
        candidate_scores: Dict[str, List[float]] = {m: [] for m in
                          ThresholdCalibrator.HIGH_METRICS + ThresholdCalibrator.LOW_METRICS}

        human_grp = CorpusGroup.HUMAN_BASELINE.value
        llm_grp   = CorpusGroup.LLM_OUTPUT.value
        train_grp = CorpusGroup.TRAIN_CORPUS.value

        for i, na in enumerate(all_names):
            for j, nb in enumerate(all_names):
                if j <= i:
                    continue
                ga, gb = name_to_group[na], name_to_group[nb]

                is_null      = (ga == human_grp and gb == human_grp)
                is_candidate = ({ga, gb} == {llm_grp, train_grp})

                if not is_null and not is_candidate:
                    continue

                scores_dict = null_scores if is_null else candidate_scores
                scores_dict["jaccard_n5"].append(float(ngram_mat[i, j]))
                scores_dict["lcs_strict"].append(float(lcs_mat[i, j]))
                scores_dict["lcs_pitch"].append(float(lcs_mat[i, j]))
                scores_dict["tfidf_dist"].append(float(tfidf_dist[i, j]))
                scores_dict["hist_dist"].append(float(hist_dist[i, j]))

        if not null_scores["jaccard_n5"]:
            log.warning("No human-baseline pairs found for calibration -- "
                        "add HUMAN_BASELINE corpus or calibration will use defaults")
        if not candidate_scores["jaccard_n5"]:
            log.warning("No llmxtrain pairs found for calibration -- "
                        "add both LLM_OUTPUT and TRAIN_CORPUS corpora")

        return calibrator.calibrate(null_scores, candidate_scores)

    # -- Targeted query: one piece vs. a reference corpus -----------------

    def query_single(self, query_tokens: List[str], query_name: str = "query"
                     ) -> List[Dict]:
        """
        Compare one piece against the entire loaded corpus.
        Returns sorted list of dicts with all three metric families
        including WAR/WAQ density score.
        Useful for quickly checking a single LLM-generated piece.
        """
        from .lcs import lcs_similarity as _lcs_sim

        flat_corpus = {}
        name_to_group: Dict[str, str] = {}
        for group, corpus in self._groups.items():
            for name, seq in corpus.items():
                flat_corpus[name] = seq
                name_to_group[name] = group.value

        results = []
        for ref_name, ref_seq in flat_corpus.items():
            ng      = multi_n_overlap(query_tokens, ref_seq, ns=self.config.ngram_ns)
            lcs     = _lcs_sim(query_tokens, ref_seq,
                               ignore_duration=self.config.lcs_ignore_duration)
            density = lcs_density_score(ref_seq, query_tokens)
            results.append({
                "reference":     ref_name,
                "group":         name_to_group[ref_name],
                **{k: round(v, 5) for k, v in ng.items()},
                "lcs_sim":       round(lcs, 5),
                "density_score": density["density_score"],
                "war":           density["war"],
                "waq":           density["waq"],
                "w_ref":         density["w_ref"],
            })

        results.sort(key=lambda x: x["lcs_sim"], reverse=True)
        return results