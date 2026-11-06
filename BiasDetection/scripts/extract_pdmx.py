"""
extract_pdmx.py
---------------
Diagnoses and extracts the PDMX archive, then filters for songs with
4 or fewer instruments, copying them into data/train_corpus/.

PDMX is distributed as a multi-part archive or sometimes as a .tar.gz
disguised with a .zip extension -- this script handles both cases.

Usage:
    python scripts/extract_pdmx.py --zip PDMX.zip
    python scripts/extract_pdmx.py --zip PDMX.zip --out data/train_corpus/
    python scripts/extract_pdmx.py --zip PDMX.zip --max-instruments 4 --dry-run
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Archive diagnosis
# ---------------------------------------------------------------------------

def diagnose(zip_path: Path) -> str:
    """
    Inspect the file header to determine what it actually is.
    Returns one of: 'zip', 'tar.gz', 'tar', 'gz', '7z', 'rar', 'unknown'
    """
    with open(zip_path, "rb") as f:
        header = f.read(8)

    signatures = {
        b"PK\x03\x04":     "zip",          # standard zip
        b"PK\x05\x06":     "zip_empty",    # empty zip
        b"PK\x07\x08":     "zip_spanned",  # multi-part zip
        b"\x1f\x8b":       "tar.gz",       # gzip (often .tar.gz)
        b"BZh":            "tar.bz2",      # bzip2
        b"\xfd7zXZ\x00":   "tar.xz",       # xz
        b"7z\xbc\xaf'\"":  "7z",
        b"Rar!":           "rar",
        b"ustar":          "tar",
    }

    for sig, fmt in signatures.items():
        if header[:len(sig)] == sig:
            return fmt

    return "unknown"


def extract_archive(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract the archive regardless of its actual format.
    Returns the path to the extracted content root.
    """
    fmt = diagnose(zip_path)
    print(f"  Detected format: {fmt}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if fmt in ("zip", "zip_empty"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    elif fmt == "zip_spanned":
        # Multi-part zip -- needs all parts present
        print("  Multi-part zip detected. Looking for sibling parts...")
        parts = sorted(zip_path.parent.glob(zip_path.stem + ".z*"))
        print(f"  Found parts: {[p.name for p in parts]}")
        # Try system unzip which handles spanned archives
        result = subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(extract_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  unzip failed: {result.stderr}")
            print("  Trying 7z...")
            subprocess.run(["7z", "x", str(zip_path), f"-o{extract_dir}"], check=True)

    elif fmt in ("tar.gz", "tar.bz2", "tar.xz", "tar"):
        print(f"  File is actually a {fmt} archive (disguised as .zip)")
        with tarfile.open(zip_path) as tf:
            tf.extractall(extract_dir)

    elif fmt == "gz":
        # Bare gzip -- might be a .tar.gz
        try:
            with tarfile.open(zip_path, "r:gz") as tf:
                tf.extractall(extract_dir)
        except tarfile.TarError:
            # Plain gzip
            import gzip
            out = extract_dir / zip_path.stem
            with gzip.open(zip_path) as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    elif fmt == "7z":
        subprocess.run(["7z", "x", str(zip_path), f"-o{extract_dir}"], check=True)

    elif fmt == "rar":
        subprocess.run(["unrar", "x", str(zip_path), str(extract_dir)], check=True)

    else:
        raise ValueError(
            f"Unknown archive format (header: {open(zip_path,'rb').read(8).hex()})\n"
            f"Try: file {zip_path}"
        )

    return extract_dir


# ---------------------------------------------------------------------------
# Instrument counting
# ---------------------------------------------------------------------------

SUPPORTED = {".mid", ".midi", ".mxl", ".xml", ".musicxml", ".abc"}

def count_instruments(filepath: Path) -> int:
    """
    Count distinct instruments/parts in a music file using music21.
    Returns -1 on parse failure.
    """
    try:
        from music21 import converter
        score = converter.parse(str(filepath))
        parts = score.parts
        return len(parts) if parts else 1
    except Exception as e:
        return -1


def count_instruments_midi_fast(filepath: Path) -> int:
    """
    Fast MIDI-only instrument count without music21.
    Reads MIDI channel/track headers directly -- much faster for large corpora.
    Falls back to music21 for non-MIDI files.
    """
    if filepath.suffix.lower() not in (".mid", ".midi"):
        return count_instruments(filepath)

    try:
        with open(filepath, "rb") as f:
            header = f.read(14)
        if header[:4] != b"MThd":
            return count_instruments(filepath)
        # Bytes 10-11 = number of tracks in the MIDI file
        n_tracks = struct.unpack(">H", header[10:12])[0]
        # Track 0 is usually tempo/meta -- subtract 1 for format-1 MIDI
        fmt = struct.unpack(">H", header[8:10])[0]
        return max(1, n_tracks - 1) if fmt == 1 else n_tracks
    except Exception:
        return count_instruments(filepath)


# ---------------------------------------------------------------------------
# Filter and copy
# ---------------------------------------------------------------------------

def filter_and_copy(source_dir: Path, dest_dir: Path,
                    max_instruments: int = 4,
                    dry_run: bool = False,
                    fast: bool = True) -> dict:
    """
    Walk source_dir, count instruments per file, copy qualifying files
    to dest_dir. Returns summary stats dict.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count_fn = count_instruments_midi_fast if fast else count_instruments

    files = [
        fp for fp in source_dir.rglob("*")
        if fp.suffix.lower() in SUPPORTED and fp.is_file()
    ]

    print(f"\n  Found {len(files)} music files in {source_dir}")
    print(f"  Filtering: max_instruments = {max_instruments}")
    print(f"  Destination: {dest_dir}")
    print(f"  Dry run: {dry_run}\n")

    stats = {"total": len(files), "kept": 0, "skipped": 0,
             "failed": 0, "instrument_counts": {}}

    for i, fp in enumerate(sorted(files), 1):
        n = count_fn(fp)

        if n == -1:
            stats["failed"] += 1
            print(f"  [{i:5d}/{len(files)}] FAIL  {fp.name}")
            continue

        stats["instrument_counts"][n] = stats["instrument_counts"].get(n, 0) + 1

        if n <= max_instruments:
            stats["kept"] += 1
            dest = dest_dir / fp.name
            # Avoid name collisions -- prefix with parent dir name
            if dest.exists():
                dest = dest_dir / f"{fp.parent.name}__{fp.name}"
            if not dry_run:
                shutil.copy2(fp, dest)
            if i % 100 == 0 or i <= 5:
                print(f"  [{i:5d}/{len(files)}] KEEP  n={n}  {fp.name[:60]}")
        else:
            stats["skipped"] += 1
            if i % 500 == 0:
                print(f"  [{i:5d}/{len(files)}] SKIP  n={n}  {fp.name[:60]}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip",  required=True,
                        help="Path to the PDMX archive (any format)")
    parser.add_argument("--out",  default="data/train_corpus",
                        help="Destination for filtered MIDI files (default: data/train_corpus)")
    parser.add_argument("--extract-to", default=None,
                        help="Where to extract the archive (default: <zip_stem>_extracted/)")
    parser.add_argument("--max-instruments", type=int, default=4,
                        help="Maximum number of instruments/parts to include (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count and report without copying files")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, run filter on --extract-to directly")
    parser.add_argument("--slow", action="store_true",
                        help="Use music21 for instrument counting (slower, more accurate for non-MIDI)")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    extract_to = Path(args.extract_to) if args.extract_to \
                 else zip_path.parent / f"{zip_path.stem}_extracted"
    dest_dir   = Path(args.out)

    # Step 1: Diagnose
    print(f"Archive: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    fmt = diagnose(zip_path)
    print(f"Format:  {fmt}")

    # Step 2: Extract
    if not args.skip_extract:
        print(f"\nExtracting to: {extract_to}")
        try:
            extract_archive(zip_path, extract_to)
            print("  Extraction complete.")
        except Exception as e:
            print(f"  Extraction failed: {e}")
            print("\n  Manual fallback options:")
            print(f"    tar xf {zip_path} -C {extract_to}")
            print(f"    7z x {zip_path} -o{extract_to}")
            print(f"    python -m zipfile -e {zip_path} {extract_to}")
            sys.exit(1)
    else:
        print(f"\nSkipping extraction, using: {extract_to}")

    # Step 3: Filter and copy
    stats = filter_and_copy(
        source_dir      = extract_to,
        dest_dir        = dest_dir,
        max_instruments = args.max_instruments,
        dry_run         = args.dry_run,
        fast            = not args.slow,
    )

    # Step 4: Summary
    print("\n" + "=" * 50)
    print("  EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"  Total files found : {stats['total']}")
    print(f"  Kept (<={args.max_instruments} instruments): {stats['kept']}")
    print(f"  Skipped           : {stats['skipped']}")
    print(f"  Failed to parse   : {stats['failed']}")
    if not args.dry_run:
        print(f"  Copied to        : {dest_dir}")
    print(f"\n  Instrument count distribution:")
    for n, cnt in sorted(stats["instrument_counts"].items()):
        bar = "#" * min(40, cnt // max(1, stats["total"] // 40))
        marker = " [ok]" if n <= args.max_instruments else "  "
        print(f"    {n:3d} parts:{marker} {cnt:6d}  {bar}")
    print("=" * 50)

    if args.dry_run:
        print("\n  Dry run -- no files copied. Remove --dry-run to execute.")


if __name__ == "__main__":
    main()
