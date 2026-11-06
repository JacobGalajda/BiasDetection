"""
scripts/dedup_llm_output.py
---------------------------
Removes duplicate ABC/MIDI pairs from a corpus directory where both formats
represent the same generated piece.

NotaGen outputs both .abc and .mid for every piece, sharing an identical stem:
    notagen_R0_C1_4_tracks_2_4_sig_E_key_6505.abc
    notagen_R0_C1_4_tracks_2_4_sig_E_key_6505.mid  <- same piece

Default behaviour: KEEP .abc, DELETE (or quarantine) .mid
  ABC is preferred because:
  - Smaller file size, faster to parse
  - Human-readable -- easier to spot-check suspicious pairs manually
  - music21 parses ABC ~40x faster than MIDI off spinning disk

Flag --prefer-midi: KEEP .mid, DELETE (or quarantine) .abc
  Use this when you need 1:1 MIDI validation:
  - Comparing against a MIDI-only reference corpus
  - Running audio synthesis for listening validation
  - Any downstream tool that only accepts MIDI

Flag --mode quarantine: moves removed files to a _dedup_removed/ subfolder
  instead of deleting. Strongly recommended for a first run.

Flag --dry-run: prints what would happen without touching anything.

Usage
-----
    # See what would be removed (safe -- no changes)
    python scripts/dedup_llm_output.py --dry-run

    # Remove MIDI duplicates, keep ABC (default)
    python scripts/dedup_llm_output.py --mode quarantine

    # Remove ABC duplicates, keep MIDI (1:1 MIDI validation)
    python scripts/dedup_llm_output.py --prefer-midi --mode quarantine

    # Target a specific directory (default: data/llm_output)
    python scripts/dedup_llm_output.py --dir data/duped --dry-run

    # Hard delete instead of quarantine (irreversible -- use with caution)
    python scripts/dedup_llm_output.py --mode delete
"""

import argparse
import logging
import shutil
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_SCRIPT_DIR   = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DIR  = _PROJECT_ROOT / "data" / "llm_output"
_QUARANTINE   = "_dedup_removed"

ABC_EXTS  = {".abc"}
MIDI_EXTS = {".mid", ".midi"}
ALL_EXTS  = ABC_EXTS | MIDI_EXTS


def find_duplicates(directory: Path, prefer_midi: bool
                    ) -> list[tuple[Path, Path]]:
    """
    Walk directory and find (keep, remove) pairs where both an ABC and a
    MIDI file share the same stem.

    Returns list of (keep_path, remove_path).
    prefer_midi=False -> keep ABC, remove MIDI
    prefer_midi=True  -> keep MIDI, remove ABC
    """
    # Group files by stem (case-insensitive)
    by_stem: dict[str, dict[str, Path]] = defaultdict(dict)
    for fp in sorted(directory.rglob("*")):
        if fp.suffix.lower() not in ALL_EXTS:
            continue
        ext_group = "abc" if fp.suffix.lower() in ABC_EXTS else "midi"
        by_stem[fp.stem.lower()][ext_group] = fp

    pairs = []
    for stem, formats in by_stem.items():
        if "abc" in formats and "midi" in formats:
            if prefer_midi:
                keep   = formats["midi"]
                remove = formats["abc"]
            else:
                keep   = formats["abc"]
                remove = formats["midi"]
            pairs.append((keep, remove))

    return pairs


def run(directory: Path, prefer_midi: bool, mode: str, dry_run: bool) -> None:
    if not directory.exists():
        log.error("Directory not found: %s", directory)
        return

    pairs = find_duplicates(directory, prefer_midi)

    keep_fmt   = "MIDI" if prefer_midi else "ABC"
    remove_fmt = "ABC"  if prefer_midi else "MIDI"

    print()
    print("=" * 60)
    print(f"  DEDUPLICATION {'(DRY RUN) ' if dry_run else ''}-- {directory.name}/")
    print("=" * 60)
    print(f"  Strategy : keep {keep_fmt}, remove {remove_fmt}")
    print(f"  Mode     : {'dry-run' if dry_run else mode}")
    print(f"  Pairs    : {len(pairs)} duplicate stems found")
    print()

    if not pairs:
        print("  Nothing to do -- no ABC/MIDI duplicate pairs found.")
        return

    quarantine_dir = directory / _QUARANTINE
    if not dry_run and mode == "quarantine":
        quarantine_dir.mkdir(exist_ok=True)

    removed = 0
    for keep, remove in sorted(pairs, key=lambda p: p[0].stem.lower()):
        action = "WOULD REMOVE" if dry_run else ("QUARANTINE" if mode == "quarantine" else "DELETE")
        print(f"  [ok] keep   {keep.name}")
        print(f"  {('~' if dry_run else '[rm]')} {action.lower()}  {remove.name}")
        print()

        if dry_run:
            removed += 1
            continue

        try:
            if mode == "quarantine":
                dest = quarantine_dir / remove.name
                shutil.move(str(remove), dest)
                log.debug("Quarantined: %s -> %s", remove.name, dest)
            elif mode == "delete":
                remove.unlink()
                log.debug("Deleted: %s", remove.name)
            removed += 1
        except Exception as exc:
            log.error("Failed on %s: %s", remove.name, exc)

    print("=" * 60)
    print(f"  {'Would remove' if dry_run else 'Removed'} : {removed} files")
    if not dry_run and mode == "quarantine":
        print(f"  Quarantine : {quarantine_dir}")
    print("=" * 60)

    if not dry_run:
        remaining = sum(
            1 for fp in directory.rglob("*")
            if fp.suffix.lower() in ALL_EXTS
            and fp.parent != quarantine_dir
        )
        print(f"\n  Files remaining in {directory.name}/ : {remaining}")
        print(f"  Re-run the pipeline -- intra-LLM duplicate flags will be gone.\n")
    else:
        total = sum(1 for fp in directory.rglob("*") if fp.suffix.lower() in ALL_EXTS)
        print(f"\n  Total files now      : {total}")
        print(f"  Total after dedup    : {total - removed}")
        print(f"\n  Run without --dry-run to apply.\n")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Deduplicate ABC/MIDI pairs in llm_output/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dir", type=Path, default=_DEFAULT_DIR, metavar="PATH",
        help=f"Directory to deduplicate (default: {_DEFAULT_DIR})",
    )
    p.add_argument(
        "--prefer-midi", action="store_true",
        help="Keep MIDI, remove ABC. Use for 1:1 MIDI validation runs.",
    )
    p.add_argument(
        "--mode", choices=["quarantine", "delete"], default="quarantine",
        help="quarantine (default): move removed files to _dedup_removed/. "
             "delete: permanently remove. Quarantine is strongly recommended.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without touching any files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        directory   = args.dir,
        prefer_midi = args.prefer_midi,
        mode        = args.mode,
        dry_run     = args.dry_run,
    )
