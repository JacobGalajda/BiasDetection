"""
ingest_maestro.py
-----------------
Copies (or symlinks) MIDI files from a MAESTRO dataset directory into
data/human_baseline/, preserving enough metadata in the filename to keep
pieces identifiable without needing the original folder structure.

Usage
-----
    # Basic -- copy all MIDIs from MAESTRO into human_baseline/
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0

    # Symlink instead of copy (saves disk space, requires src to stay in place)
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0 --mode symlink

    # Dry run -- see what would happen without touching any files
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0 --dry-run

    # Filter to specific composers (comma-separated, case-insensitive substring match)
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0 --composers "beethoven,schubert"

    # Cap the number of files (useful for a quick test run)
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0 --limit 50

    # Add string quartet MIDIs from a separate folder alongside MAESTRO
    python scripts/ingest_maestro.py --src /path/to/maestro-v3.0.0 \\
        --extra /path/to/string_quartets

MAESTRO structure
-----------------
MAESTRO ships in two layouts -- both are handled:

  Layout A (v2 / v3, year-based subdirs):
    maestro-v3.0.0/
      2004/  2006/  2008/ ... 2018/
        *.midi

  Layout B (flat):
    maestro-v3.0.0/
      *.midi  (or *.mid)

The script also reads maestro-v3.0.0/maestro-v3.0.0.csv (or .json) if
present to extract composer names for filtering. If the metadata file is
absent, all MIDIs are ingested without filtering.

Output filenames
----------------
Files are renamed to avoid collisions when flattening the year subdirs:

    {year}_{original_stem}.mid

e.g.  2004_beethoven_op57.mid

If no year subdir is detected, the original filename is preserved.
"""

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults -- paths relative to the project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DEST = _PROJECT_ROOT / "data" / "human_baseline"

MIDI_EXTENSIONS = {".mid", ".midi"}


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _load_maestro_metadata(src: Path) -> dict:
    """
    Try to load composer/title metadata from the MAESTRO CSV or JSON file.
    Returns {midi_filename: {"composer": str, "title": str}} or {} if absent.
    """
    # Try CSV first (v3), then JSON (v2)
    for pattern in ["*.csv", "*.json"]:
        candidates = list(src.glob(pattern))
        if not candidates:
            continue
        meta_file = candidates[0]
        log.info("Loading metadata from %s", meta_file.name)

        if meta_file.suffix == ".csv":
            meta = {}
            with open(meta_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = Path(row.get("midi_filename", "")).name
                    meta[fname] = {
                        "composer": row.get("canonical_composer", ""),
                        "title":    row.get("canonical_title", ""),
                        "year":     row.get("year", ""),
                    }
            return meta

        if meta_file.suffix == ".json":
            with open(meta_file, encoding="utf-8") as f:
                raw = json.load(f)
            meta = {}
            # MAESTRO JSON: keys are column names, values are dicts of index->value
            if "midi_filename" in raw:
                indices = raw["midi_filename"].keys()
                for idx in indices:
                    fname = Path(raw["midi_filename"][idx]).name
                    meta[fname] = {
                        "composer": raw.get("canonical_composer", {}).get(idx, ""),
                        "title":    raw.get("canonical_title",    {}).get(idx, ""),
                        "year":     str(raw.get("year",           {}).get(idx, "")),
                    }
            return meta

    log.warning("No MAESTRO metadata file found -- ingesting all MIDIs without filtering")
    return {}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_midi_files(directory: Path) -> list[tuple[Path, str]]:
    """
    Recursively find MIDI files.  Returns list of (path, year_prefix) where
    year_prefix is the immediate parent directory name if it looks like a year
    (4-digit number), otherwise "".
    """
    results = []
    for fp in sorted(directory.rglob("*")):
        if fp.suffix.lower() not in MIDI_EXTENSIONS:
            continue
        parent = fp.parent.name
        year   = parent if (parent.isdigit() and len(parent) == 4) else ""
        results.append((fp, year))
    return results


# ---------------------------------------------------------------------------
# Composer filter
# ---------------------------------------------------------------------------

def _matches_composer(meta_entry: dict, composers: list[str]) -> bool:
    if not composers:
        return True
    composer_lower = meta_entry.get("composer", "").lower()
    return any(c in composer_lower for c in composers)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(src: Path, dest: Path, extra: list[Path],
           mode: str, composers: list[str], limit: int,
           dry_run: bool) -> None:
    """
    Main ingestion logic.

    mode : "copy" | "symlink" | "move"
    """
    dest.mkdir(parents=True, exist_ok=True)
    metadata = _load_maestro_metadata(src)

    # Collect files from primary source
    all_files: list[tuple[Path, str]] = _find_midi_files(src)

    # Collect files from extra directories (e.g. string quartets)
    for extra_dir in extra:
        if not extra_dir.exists():
            log.warning("Extra directory not found, skipping: %s", extra_dir)
            continue
        extra_files = _find_midi_files(extra_dir)
        log.info("Found %d MIDI files in extra dir: %s", len(extra_files), extra_dir)
        all_files.extend(extra_files)

    log.info("Total MIDI files found: %d", len(all_files))

    # Filter by composer
    if composers and metadata:
        filtered = []
        for fp, year in all_files:
            entry = metadata.get(fp.name, {})
            if _matches_composer(entry, composers):
                filtered.append((fp, year))
        log.info("After composer filter (%s): %d files", ", ".join(composers), len(filtered))
        all_files = filtered
    elif composers and not metadata:
        log.warning("--composers flag set but no metadata file found; ignoring filter")

    # Apply limit
    if limit and limit < len(all_files):
        log.info("Applying --limit: keeping %d of %d files", limit, len(all_files))
        all_files = all_files[:limit]

    # Ingest
    copied = skipped = failed = 0

    for fp, year in all_files:
        # Build output filename
        stem   = fp.stem
        suffix = ".mid"   # normalize .midi -> .mid
        if year:
            out_name = f"{year}_{stem}{suffix}"
        else:
            out_name = f"{stem}{suffix}"

        dest_path = dest / out_name

        # Handle collisions
        if dest_path.exists():
            log.debug("Skipping (already exists): %s", out_name)
            skipped += 1
            continue

        meta_entry = metadata.get(fp.name, {})
        composer   = meta_entry.get("composer", "unknown")
        title      = meta_entry.get("title",    fp.stem)

        action_str = f"{mode.upper():7s}  {fp.name}  ->  {out_name}"
        if meta_entry:
            action_str += f"  [{composer} -- {title}]"

        if dry_run:
            print(f"  DRY RUN  {action_str}")
            copied += 1
            continue

        try:
            if mode == "copy":
                shutil.copy2(fp, dest_path)
            elif mode == "symlink":
                dest_path.symlink_to(fp.resolve())
            elif mode == "move":
                shutil.move(str(fp), dest_path)
            log.debug(action_str)
            copied += 1
        except Exception as exc:
            log.error("Failed on %s: %s", fp.name, exc)
            failed += 1

    # Summary
    print()
    print("=" * 55)
    print(f"  MAESTRO INGEST {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print("=" * 55)
    print(f"  {'Would process' if dry_run else 'Processed'} : {copied}")
    print(f"  Already existed (skipped) : {skipped}")
    print(f"  Failed                    : {failed}")
    print(f"  Destination               : {dest}")
    print("=" * 55)

    if failed:
        log.warning("%d files failed -- check logs above", failed)
    if not dry_run and copied:
        print(f"\n  Next step: run the pipeline")
        print(f"  python experiments/run_full_analysis.py\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Ingest MAESTRO MIDI files into data/human_baseline/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--src", required=True, type=Path,
        metavar="PATH",
        help="Path to the MAESTRO dataset root directory",
    )
    p.add_argument(
        "--dest", type=Path, default=_DEFAULT_DEST,
        metavar="PATH",
        help=f"Destination directory (default: {_DEFAULT_DEST})",
    )
    p.add_argument(
        "--extra", type=Path, nargs="*", default=[],
        metavar="PATH",
        help="Additional directories to include (e.g. string quartet MIDIs)",
    )
    p.add_argument(
        "--mode", choices=["copy", "symlink", "move"], default="copy",
        help="File operation mode (default: copy)",
    )
    p.add_argument(
        "--composers", type=str, default="",
        metavar="NAMES",
        help='Comma-separated composer substrings to filter by, e.g. "beethoven,schubert"',
    )
    p.add_argument(
        "--limit", type=int, default=0,
        metavar="N",
        help="Maximum number of files to ingest (0 = no limit)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without touching any files",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show each file as it is processed",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    composers = [c.strip().lower() for c in args.composers.split(",") if c.strip()]

    ingest(
        src       = args.src,
        dest      = args.dest,
        extra     = args.extra or [],
        mode      = args.mode,
        composers = composers,
        limit     = args.limit,
        dry_run   = args.dry_run,
    )
