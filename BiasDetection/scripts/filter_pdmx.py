"""
filter_pdmx.py
--------------
Reads PDMX.csv, filters by instrument count and quality, and copies
.mid files into output directories.

Supports a two-way split for the robust null hypothesis design:
  --sq-split N   reserves N deduplicated 4-track pieces as a separate
                 human_baseline_sq/ directory, then excludes those exact
                 files from train_corpus/ so the sets are non-overlapping.

Full example for the dual-baseline design:
    python scripts/filter_pdmx.py \\
        --pdmx ../PDMX_extracted/PDMX \\
        --csv  ../PDMX_extracted/PDMX/PDMX_v9.csv \\
        --max-instruments 4 \\
        --deduplicated \\
        --min-notes 100 \\
        --sq-split 500 \\
        --sq-out  data/human_baseline_sq \\
        --out     data/train_corpus \\
        --dry-run
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def resolve_path(raw: str, pdmx_dir: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else pdmx_dir / p


def find_mid(row, pdmx_dir: Path, has_mid_col: bool) -> Path | None:
    if has_mid_col:
        val = row.get("mid", None)
        if pd.isna(val) or not val:
            return None
        p = resolve_path(str(val), pdmx_dir)
        return p if p.exists() else None
    else:
        try:
            json_path = resolve_path(str(row["path"]), pdmx_dir)
            mid_path  = (pdmx_dir / "mid" /
                         json_path.relative_to(pdmx_dir / "data")).with_suffix(".mid")
            if mid_path.exists():
                return mid_path
            mxl_path  = (pdmx_dir / "mxl" /
                         json_path.relative_to(pdmx_dir / "data")).with_suffix(".mxl")
            return mxl_path if mxl_path.exists() else None
        except ValueError:
            return None


def print_stats(df: pd.DataFrame, label: str):
    print(f"\n  [{label}]  n={len(df):,}")
    if "n_notes" in df.columns:
        print(f"    n_notes  mean={df['n_notes'].mean():.0f}  "
              f"median={df['n_notes'].median():.0f}  "
              f"p5={df['n_notes'].quantile(0.05):.0f}  "
              f"p95={df['n_notes'].quantile(0.95):.0f}")
    print(f"    n_tracks distribution:")
    for n, cnt in df["n_tracks"].value_counts().sort_index().items():
        print(f"      {n} tracks: {cnt:,}")


def copy_files(df: pd.DataFrame, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, music_path in enumerate(df["music_path"], 1):
        dest = out_dir / music_path.name
        if dest.exists():
            dest = out_dir / f"{music_path.parent.name}__{music_path.name}"
        shutil.copy2(music_path, dest)
        copied += 1
        if i % 5000 == 0 or i == len(df):
            print(f"  [{label}] {i:,} / {len(df):,} copied ...")
    print(f"  [{label}] Done -- {copied:,} files -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdmx", required=True,
                        help="Path to the extracted PDMX/ directory")
    parser.add_argument("--csv", default=None,
                        help="Override CSV path (use PDMX_v9.csv if available)")
    parser.add_argument("--out", default="data/train_corpus",
                        help="train_corpus output directory (default: data/train_corpus)")
    parser.add_argument("--max-instruments", type=int, default=4,
                        help="Maximum n_tracks for train_corpus (default: 4)")
    parser.add_argument("--deduplicated", action="store_true",
                        help="Only include subset:deduplicated == True")
    parser.add_argument("--min-notes", type=int, default=100,
                        help="Minimum n_notes (default: 100)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Copy at most this many files to train_corpus (0=no limit)")

    # SQ split arguments
    parser.add_argument("--sq-split", type=int, default=0,
                        help="Reserve this many n_tracks==4 pieces as "
                             "human_baseline_sq (non-overlapping with train_corpus). "
                             "0 = no split.")
    parser.add_argument("--sq-out", default="data/human_baseline_sq",
                        help="human_baseline_sq output directory "
                             "(default: data/human_baseline_sq)")
    parser.add_argument("--sq-seed", type=int, default=42,
                        help="Random seed for SQ split (default: 42)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without copying files")
    args = parser.parse_args()

    pdmx_dir = Path(args.pdmx)
    csv_path = Path(args.csv) if args.csv else pdmx_dir / "PDMX.csv"

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        raise SystemExit(1)

    # -- Load -----------------------------------------------------------------
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    has_mid_col = "mid" in df.columns
    print(f"Total songs: {len(df):,}  |  CSV: {'v9+' if has_mid_col else 'v1 fallback'}")

    # -- Base filters (applied to everything) ---------------------------------
    stages = []

    if args.deduplicated:
        col = "subset:deduplicated"
        if col in df.columns:
            df = df[df[col] == True].copy()
            stages.append(f"deduplicated: {len(df):,}")
        else:
            print(f"WARNING: '{col}' column not found -- skipping deduplication")

    if args.min_notes > 0 and "n_notes" in df.columns:
        df = df[df["n_notes"] >= args.min_notes].copy()
        stages.append(f"min_notes>={args.min_notes}: {len(df):,}")

    for s in stages:
        print(f"  After {s} songs remaining")

    # -- SQ split (n_tracks == 4 only) ----------------------------------------
    sq_df = pd.DataFrame()
    sq_indices = set()

    if args.sq_split > 0:
        sq_pool = df[df["n_tracks"] == 4].copy()
        if len(sq_pool) < args.sq_split:
            print(f"WARNING: requested --sq-split {args.sq_split} but only "
                  f"{len(sq_pool)} 4-track pieces available. Using all of them.")
            sq_df = sq_pool
        else:
            sq_df = sq_pool.sample(n=args.sq_split, random_state=args.sq_seed)
        sq_indices = set(sq_df.index)
        print(f"\nSQ split: {len(sq_df):,} pieces reserved for human_baseline_sq")

    # -- Train corpus (excluding SQ split pieces) ------------------------------
    train_df = df[
        (df["n_tracks"] <= args.max_instruments) &
        (~df.index.isin(sq_indices))
    ].copy()

    if args.limit > 0:
        train_df = train_df.head(args.limit)

    # -- Resolve file paths ----------------------------------------------------
    print("\nResolving file paths ...")
    if not sq_df.empty:
        sq_df["music_path"] = sq_df.apply(
            lambda r: find_mid(r, pdmx_dir, has_mid_col), axis=1)
        sq_df = sq_df[sq_df["music_path"].notna()]

    train_df["music_path"] = train_df.apply(
        lambda r: find_mid(r, pdmx_dir, has_mid_col), axis=1)
    train_df = train_df[train_df["music_path"].notna()]

    # -- Stats -----------------------------------------------------------------
    print("\n-- Summary ----------------------------------------------------------")
    if not sq_df.empty:
        print_stats(sq_df,    "human_baseline_sq  (n_tracks==4, for matching null)")
    print_stats(train_df,     "train_corpus       (n_tracks<=4, SQ excluded)")

    # Verify non-overlap
    if not sq_df.empty:
        sq_names   = set(sq_df["music_path"].apply(lambda p: p.name))
        train_names = set(train_df["music_path"].apply(lambda p: p.name))
        overlap = sq_names & train_names
        print(f"\nOverlap check: {len(overlap)} files in both sets "
              f"({'CLEAN' if len(overlap) == 0 else 'WARNING: FIX NEEDED'})")

    if args.dry_run:
        print("\nDry run -- no files copied.")
        print("Remove --dry-run to execute.")
        return

    # -- Copy -----------------------------------------------------------------
    if not sq_df.empty:
        copy_files(sq_df,   Path(args.sq_out),  "human_baseline_sq")
    copy_files(train_df, Path(args.out), "train_corpus")


if __name__ == "__main__":
    main()
