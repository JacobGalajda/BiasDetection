"""
preconvert_abc.py
-----------------
Pre-converts all ABC files in a directory to MIDI using music21,
with a hard per-file timeout so malformed files never stall the batch.

Run this ONCE on your llm_output directory. Future pipeline runs will
load the .mid files instead of parsing ABC, which is ~10x faster and
never hangs.

Skipped files are written to a skip-list so you can inspect them.

Usage:
    python scripts/preconvert_abc.py --dir data/llm_output
    python scripts/preconvert_abc.py --dir data/llm_output --timeout 30
    python scripts/preconvert_abc.py --dir data/llm_output --dry-run
"""

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")


def _convert_worker(abc_path: str, out_path: str, result_q: mp.Queue):
    """Runs in a subprocess -- safe to hard-kill on timeout."""
    try:
        from music21 import converter
        score = converter.parse(abc_path)
        score.write("midi", fp=out_path)
        result_q.put(("ok", None))
    except Exception as exc:
        result_q.put(("error", str(exc)))


def convert_file(abc_path: Path, timeout: int) -> tuple[str, str]:
    """
    Convert one ABC file to MIDI in a subprocess with a hard timeout.
    Returns ("ok"|"timeout"|"error"|"exists", message).
    """
    out_path = abc_path.with_suffix(".mid")
    if out_path.exists():
        return "exists", str(out_path)

    q = mp.Queue()
    p = mp.Process(target=_convert_worker,
                   args=(str(abc_path), str(out_path), q),
                   daemon=True)
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        return "timeout", f"killed after {timeout}s"

    try:
        status, msg = q.get_nowait()
        return status, msg or str(out_path)
    except Exception:
        return "error", "no result from worker"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True,
                        help="Directory containing .abc files")
    parser.add_argument("--timeout", type=int, default=25,
                        help="Hard timeout per file in seconds (default: 25)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without converting")
    args = parser.parse_args()

    root  = Path(args.dir)
    files = sorted(root.rglob("*.abc"))
    print(f"Found {len(files):,} .abc files in {root}")

    if args.dry_run:
        already = sum(1 for f in files if f.with_suffix(".mid").exists())
        print(f"  Already converted : {already:,}")
        print(f"  To convert        : {len(files) - already:,}")
        return

    stats  = {"ok": 0, "exists": 0, "timeout": 0, "error": 0}
    skipped = []
    t_start = time.time()

    try:
        from tqdm import tqdm
        it = tqdm(files, unit="file", dynamic_ncols=True)
    except ImportError:
        it = files

    for fp in it:
        status, msg = convert_file(fp, args.timeout)
        stats[status] = stats.get(status, 0) + 1

        if status == "timeout":
            logging.warning("TIMEOUT: %s", fp.name)
            skipped.append((fp.name, "timeout"))
        elif status == "error":
            logging.warning("ERROR: %s -- %s", fp.name, msg)
            skipped.append((fp.name, msg))

        if hasattr(it, "set_postfix"):
            it.set_postfix(stats)

    elapsed = time.time() - t_start
    print(f"\n-- Conversion complete in {elapsed:.0f}s --")
    print(f"  Converted  : {stats['ok']:,}")
    print(f"  Pre-existed: {stats['exists']:,}")
    print(f"  Timeouts   : {stats['timeout']:,}")
    print(f"  Errors     : {stats['error']:,}")

    if skipped:
        skip_path = root / "_skipped_files.txt"
        with open(skip_path, "w") as f:
            for name, reason in skipped:
                f.write(f"{name}\t{reason}\n")
        print(f"\nSkip list saved to: {skip_path}")
        print("These files will also be skipped automatically by pipeline_fast.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()
