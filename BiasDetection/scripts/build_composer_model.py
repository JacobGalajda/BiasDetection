"""
build_composer_model.py
-----------------------
Tokenize MAESTRO MIDI files, build TF-IDF composer centroids,
perform train/test split, and save the composer model for use
by visualize_composer_space.py and classify_composer.py.

Usage
-----
    python scripts/build_composer_model.py \
        --maestro_root  ../maestro-v3.0.0 \
        --out           results/composer/composer_model.json \
        [--min_pieces   30] \
        [--test_frac    0.2] \
        [--use_interval]     # use interval tokens instead of abs-pitch
        [--workers      4]

Output
------
    results/composer/composer_model.json  -- centroids, vocab, train/test split
    results/composer/tokens/              -- cached token files per piece
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from music_bias_analyzer.tokenizer import file_to_tokens
except ImportError as e:
    print(f"[error] {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# Interval tokenization (reused from MCIC/SMP scripts)
# ---------------------------------------------------------------------------
def to_interval_tokens(abs_tokens):
    import re as _re
    NOTE_TO_MIDI = {
        "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
        "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,
    }
    def tok_to_midi(tok):
        m = _re.match(r'^([A-Ga-g][#b]?)(-?\d+)', tok.split("_")[0])
        if not m: return None
        note, oct_ = m.group(1).capitalize(), int(m.group(2))
        s = NOTE_TO_MIDI.get(note)
        return None if s is None else s + (oct_ + 1) * 12
    out, prev = [], None
    for tok in abs_tokens:
        parts = tok.split("_", 1)
        dur = parts[1] if len(parts) > 1 else "unknown"
        midi = tok_to_midi(tok)
        out.append(f"0_{dur}" if (midi is None or prev is None)
                   else f"{midi-prev}_{dur}")
        if midi is not None: prev = midi
    return out

# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------
def build_tfidf(token_lists: list[list[str]]) -> tuple[dict, list[dict]]:
    """
    Build TF-IDF vectors from a list of token sequences.
    Returns (idf_dict, list_of_tfidf_dicts).
    """
    N = len(token_lists)
    # Document frequency
    df = Counter()
    for tokens in token_lists:
        df.update(set(tokens))
    idf = {tok: math.log((N + 1) / (df[tok] + 1)) + 1 for tok in df}

    tfidf_vecs = []
    for tokens in token_lists:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = {tok: (tf[tok] / total) * idf[tok] for tok in tf}
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        tfidf_vecs.append({tok: v/norm for tok, v in vec.items()})
    return idf, tfidf_vecs

def centroid(vecs: list[dict]) -> dict:
    """Average a list of TF-IDF vectors into a centroid."""
    c: dict = defaultdict(float)
    for v in vecs:
        for tok, val in v.items():
            c[tok] += val / len(vecs)
    return dict(c)

def cosine(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    na  = math.sqrt(sum(v*v for v in a.values())) or 1e-9
    nb  = math.sqrt(sum(v*v for v in b.values())) or 1e-9
    return dot / (na * nb)

# ---------------------------------------------------------------------------
# Worker: tokenize one file
# ---------------------------------------------------------------------------
def tokenize_file(args):
    piece_id, midi_path, use_interval, cache_dir = args
    cache_path = cache_dir / f"{piece_id}.json"
    if cache_path.exists():
        return piece_id, json.load(open(cache_path))
    try:
        toks = file_to_tokens(str(midi_path))
        if use_interval:
            toks = to_interval_tokens(toks)
        json.dump(toks, open(cache_path, "w"))
        return piece_id, toks
    except Exception as e:
        return piece_id, []

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maestro_root", default="../maestro-v3.0.0")
    ap.add_argument("--out",          default="results/composer/composer_model.json")
    ap.add_argument("--min_pieces",   type=int,   default=30,
                    help="Minimum pieces per composer to include")
    ap.add_argument("--composers",    type=str,   default=None,
                    help="Comma-separated list of composers to include "
                         "(overrides --min_pieces filter). "
                         "Example: 'Bach,Beethoven,Chopin,Liszt,Schubert'")
    ap.add_argument("--test_frac",    type=float, default=0.2)
    ap.add_argument("--use_interval", action="store_true",
                    help="Use interval tokens instead of absolute-pitch")
    ap.add_argument("--workers",      type=int,
                    default=max(1, mp.cpu_count()-1))
    args = ap.parse_args()

    root     = Path(args.maestro_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path.parent / "tokens"
    cache_dir.mkdir(exist_ok=True)

    token_mode = "interval" if args.use_interval else "absolute"
    print(f"\nMAESTRO root:  {root}")
    print(f"Token mode:    {token_mode}")
    print(f"Min pieces:    {args.min_pieces}")
    print(f"Test fraction: {args.test_frac}")

    # -- Load metadata ------------------------------------------------------
    meta = json.load(open(root / "maestro-v3.0.0.json"))
    ids      = list(meta["canonical_composer"].keys())
    composers_raw = {i: meta["canonical_composer"][i] for i in ids}
    midi_paths    = {i: root / meta["midi_filename"][i] for i in ids}

    # Filter: exclude joint/arrangement entries and low-count composers
    EXCLUDE = {"/", "&"}
    composer_counts = Counter(composers_raw.values())
    valid_composers = {c for c, n in composer_counts.items()
                       if n >= args.min_pieces
                       and not any(x in c for x in EXCLUDE)}

    # Optional: restrict to specific composers by last name or full name
    if args.composers:
        requested = [c.strip() for c in args.composers.split(",")]
        def matches(comp_name):
            return any(
                req.lower() in comp_name.lower() or
                comp_name.lower().endswith(req.lower())
                for req in requested
            )
        valid_composers = {c for c in valid_composers if matches(c)}
        print(f"Restricting to {len(valid_composers)} requested composers: "
              f"{sorted(valid_composers)}")

    pieces = [(i, composers_raw[i], midi_paths[i])
              for i in ids
              if composers_raw[i] in valid_composers]

    print(f"\nValid composers: {len(valid_composers)}")
    print(f"Valid pieces:    {len(pieces)}")
    print("\nComposer distribution:")
    dist = Counter(c for _, c, _ in pieces)
    for comp, n in dist.most_common():
        print(f"  {n:>4}  {comp}")

    # -- Train/test split (stratified per composer) -------------------------
    import random; random.seed(42)
    by_composer = defaultdict(list)
    for pid, comp, path in pieces:
        by_composer[comp].append((pid, path))

    train_ids, test_ids = set(), set()
    for comp, items in by_composer.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * args.test_frac))
        for pid, _ in items[:n_test]:
            test_ids.add(pid)
        for pid, _ in items[n_test:]:
            train_ids.add(pid)

    print(f"\nTrain pieces: {len(train_ids)}")
    print(f"Test pieces:  {len(test_ids)}")

    # -- Tokenize all pieces ------------------------------------------------
    print(f"\nTokenizing {len(pieces)} pieces with {args.workers} workers ...")
    worker_args = [
        (pid, path, args.use_interval,
         cache_dir / ("int" if args.use_interval else "abs"))
        for pid, _, path in pieces
    ]
    (cache_dir / ("int" if args.use_interval else "abs")).mkdir(exist_ok=True)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("  (install tqdm for progress bar: pip install tqdm)")

    if args.workers == 1:
        if has_tqdm:
            results = [tokenize_file(a) for a in tqdm(worker_args,
                       desc="Tokenizing", unit="piece")]
        else:
            results = []
            for i, a in enumerate(worker_args):
                results.append(tokenize_file(a))
                if (i+1) % 50 == 0 or (i+1) == len(worker_args):
                    print(f"  {i+1}/{len(worker_args)} pieces tokenized")
    else:
        with mp.Pool(args.workers) as pool:
            if has_tqdm:
                results = list(tqdm(
                    pool.imap(tokenize_file, worker_args),
                    total=len(worker_args),
                    desc="Tokenizing",
                    unit="piece"
                ))
            else:
                results = pool.map(tokenize_file, worker_args)

    token_map = {pid: toks for pid, toks in results if toks}
    empty = len(pieces) - len(token_map)
    print(f"  Tokenized: {len(token_map)}  Empty/failed: {empty}")

    # -- Build TF-IDF per composer (train set only) -------------------------
    print("\nBuilding TF-IDF composer centroids ...")
    comp_train_tokens = defaultdict(list)
    for pid, comp, _ in pieces:
        if pid in train_ids and pid in token_map:
            comp_train_tokens[comp].append(token_map[pid])

    # Global IDF from all training tokens
    all_train_tokens = [t for tl in comp_train_tokens.values() for t in tl]
    global_idf, _ = build_tfidf(all_train_tokens)

    # Per-piece TF-IDF vectors using global IDF
    def tfidf_vec(tokens):
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = {tok: (tf[tok]/total) * global_idf.get(tok, 1.0) for tok in tf}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {tok: v/norm for tok, v in vec.items()}

    # Composer centroids
    comp_centroids = {}
    comp_piece_vecs = {}  # for visualization
    for comp, token_lists in comp_train_tokens.items():
        vecs = [tfidf_vec(tl) for tl in token_lists]
        comp_centroids[comp]  = centroid(vecs)
        comp_piece_vecs[comp] = vecs

    # -- Evaluate on test set -----------------------------------------------
    print("\nEvaluating on test set ...")
    correct_top1 = 0
    correct_top3 = 0
    n_test_scored = 0
    per_composer_results = defaultdict(lambda: {"correct":0,"total":0})

    for pid, comp, _ in pieces:
        if pid not in test_ids or pid not in token_map:
            continue
        vec = tfidf_vec(token_map[pid])
        scores = {c: cosine(vec, cen)
                  for c, cen in comp_centroids.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)

        n_test_scored += 1
        per_composer_results[comp]["total"] += 1
        if ranked[0] == comp:
            correct_top1 += 1
            per_composer_results[comp]["correct"] += 1
        if comp in ranked[:3]:
            correct_top3 += 1

    acc_top1 = correct_top1 / n_test_scored if n_test_scored else 0
    acc_top3 = correct_top3 / n_test_scored if n_test_scored else 0

    print(f"\n{'='*50}")
    print(f"COMPOSER CLASSIFICATION RESULTS ({token_mode} tokens)")
    print(f"{'='*50}")
    print(f"Top-1 accuracy: {acc_top1:.3f}  ({correct_top1}/{n_test_scored})")
    print(f"Top-3 accuracy: {acc_top3:.3f}  ({correct_top3}/{n_test_scored})")
    # Balanced accuracy = mean per-class recall
    per_class_recall = [
        r["correct"] / max(r["total"], 1)
        for r in per_composer_results.values()
    ]
    balanced_acc = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0

    print(f"Balanced accuracy: {balanced_acc:.3f}")
    print(f"\nPer-composer Top-1:")
    for comp, r in sorted(per_composer_results.items(),
                          key=lambda x: -x[1]["correct"]/max(x[1]["total"],1)):
        acc = r["correct"] / max(r["total"], 1)
        print(f"  {acc:.3f}  {r['correct']:>2}/{r['total']:>2}  {comp}")

    # -- Save model ---------------------------------------------------------
    model = {
        "token_mode":    token_mode,
        "min_pieces":    args.min_pieces,
        "test_frac":     args.test_frac,
        "composers":     list(comp_centroids.keys()),
        "centroids":     comp_centroids,
        "global_idf":    global_idf,
        "train_ids":     list(train_ids),
        "test_ids":      list(test_ids),
        "piece_meta":    {pid: {"composer": comp,
                                "split": "test" if pid in test_ids else "train",
                                "midi": str(path)}
                          for pid, comp, path in pieces
                          if pid in token_map},
        "results": {
            "top1_accuracy":    round(acc_top1,      4),
            "top3_accuracy":    round(acc_top3,      4),
            "balanced_accuracy": round(balanced_acc, 4),
            "n_test":           n_test_scored,
        }
    }

    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {out_path}")
    print(f"Next: python scripts/visualize_composer_space.py --model {out_path}")

if __name__ == "__main__":
    main()