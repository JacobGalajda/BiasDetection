"""
ingest_mcic.py
--------------
Ingest MCIC MIDI files into data/mcic/ and generate labels.csv.

The MCIC spreadsheet (from https://tinyurl.com/mcic-list) must be
exported as CSV first. Download from:
    https://docs.google.com/spreadsheets/d/1dzVled_Zb813IGAzZ-qO2uJJpkGbb-MSfvULyLiIdh8
    File -> Download -> CSV

Usage:
    python scripts/ingest_mcic.py \
        --midi_dir  "MCIC/MCIC DATA/MIDI FILES" \
        --meta_csv  "MCIC/mcic_metadata.csv" \
        --out_dir   "data/mcic"

Output:
    data/mcic/midi/          -- copies of all 232 MIDI files
    data/mcic/labels.csv     -- one row per case with outcome label

labels.csv columns:
    case_id       : zero-padded integer (001-116)
    file_a        : relative path to complainant MIDI
    file_b        : relative path to defendant MIDI
    title_a       : complainant title from filename
    title_b       : defendant title from filename
    outcome       : Infringed | Denied | Settled  (from metadata CSV)
    outcome_bin   : 1 = Infringed, 0 = Denied, -1 = Settled (excluded)

If --meta_csv is not provided, outcome columns are left blank and must
be filled manually from the MCIC list spreadsheet.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_midi_dir(midi_dir: Path) -> dict:
    """
    Returns {case_id: {"a": Path, "b": Path}} for all 232 files.
    Handles the NNN_a_Title.mid / NNN_b_Title.mid convention.
    """
    cases = {}
    for f in sorted(midi_dir.glob("*.mid")):
        m = re.match(r'^(\d+)_([ab])_(.+)\.mid$', f.name, re.IGNORECASE)
        if not m:
            # Fallback: some files may use _1/_2 instead of _a/_b
            m = re.match(r'^(\d+)_([12])_(.+)\.mid$', f.name, re.IGNORECASE)
            if m:
                side = "a" if m.group(2) == "1" else "b"
            else:
                print(f"  [warn] unrecognized filename: {f.name}")
                continue
        else:
            side = m.group(2).lower()

        case_id = m.group(1).zfill(3)
        title   = m.group(3).replace("_", " ")

        if case_id not in cases:
            cases[case_id] = {}
        cases[case_id][side]       = f
        cases[case_id][f"title_{side}"] = title

    return cases


def load_metadata(meta_csv: Path) -> dict:
    """
    Load outcome labels from the exported MCIC spreadsheet CSV.
    Expects columns: Case No., Court Decision (or similar).
    Returns {case_id: outcome_string}.
    """
    if not meta_csv or not meta_csv.exists():
        return {}

    outcomes = {}
    with open(meta_csv, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f, dialect="excel")
        headers = reader.fieldnames or []
        print(f"  Metadata CSV headers: {headers}")

        # Flexible column name matching
        # MCIC spreadsheet uses 'Case' and 'Decision' exactly
        id_col  = next((h for h in headers if h.strip() == "Case"), None)
        if not id_col:
            id_col = next((h for h in headers if "case" in h.lower() and
                           ("no" in h.lower() or "id" in h.lower() or "#" in h or
                            h.strip().lower() == "case")), None)
        out_col = next((h for h in headers if h.strip() == "Decision"), None)
        if not out_col:
            out_col = next((h for h in headers if "decision" in h.lower() or
                            "outcome" in h.lower() or "verdict" in h.lower() or
                            "result" in h.lower()), None)

        if not id_col or not out_col:
            print(f"  [warn] Could not identify case ID or outcome column.")
            print(f"         Available headers: {headers}")
            print(f"         Set id_col/out_col manually in this script.")
            return {}

        print(f"  Using: case_id='{id_col}', outcome='{out_col}'")

        for i, row in enumerate(reader):
            raw_id  = row.get(id_col, "").strip()
            outcome = row.get(out_col, "").strip()

            if not raw_id:
                continue

            # Case IDs are stored as "Case 1", "Case 2", etc.
            # Strip the "Case " prefix and parse the integer
            import re as _re
            m = _re.search(r'(\d+)', raw_id)
            if not m:
                continue
            case_id = str(int(m.group(1))).zfill(3)

            # Each case has two rows (complainant + defendant)
            # Only record if outcome is non-empty or not yet seen
            if case_id not in outcomes or outcome:
                outcomes[case_id] = normalize_outcome(outcome)

    return outcomes


def normalize_outcome(raw: str) -> str:
    """Map MCIC decision values to canonical Infringed / Denied / Settled.

    MCIC spreadsheet convention:
        "1"  -> Infringed (plagiarism found)
        "0"  -> Denied    (no plagiarism)
        ""   -> Settled / undecided (blank)
    Also handles legacy text spellings for robustness.
    """
    r = raw.strip()

    # Numeric MCIC convention
    if r == "1":  return "Infringed"
    if r == "0":  return "Denied"
    if r == "":   return "Settled"

    # Text fallback
    rl = r.lower()
    if any(w in rl for w in ["infring", "liable", "guilty", "found for plaintiff"]):
        return "Infringed"
    if any(w in rl for w in ["denied", "dismiss", "not infring", "found for defendant", "no infring"]):
        return "Denied"
    if any(w in rl for w in ["settl", "withdrawn", "consent"]):
        return "Settled"
    return r.title()

def outcome_bin(outcome: str) -> str:
    """1 = Infringed, 0 = Denied, -1 = Settled/Unknown."""
    if outcome == "Infringed": return "1"
    if outcome == "Denied":    return "0"
    return "-1"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--midi_dir",  required=True,
                    help="Path to MCIC/MCIC DATA/MIDI FILES directory")
    ap.add_argument("--meta_csv",  default=None,
                    help="Path to exported MCIC metadata CSV (optional)")
    ap.add_argument("--out_dir",   default="data/mcic",
                    help="Output directory (default: data/mcic)")
    ap.add_argument("--copy",      action="store_true",
                    help="Copy MIDI files into out_dir/midi/ (default: False, just generate labels)")
    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    out_dir  = Path(args.out_dir)
    meta_csv = Path(args.meta_csv) if args.meta_csv else None

    # --- parse MIDI filenames ---
    print(f"\nParsing MIDI files from {midi_dir} ...")
    cases = parse_midi_dir(midi_dir)
    print(f"  Found {len(cases)} cases, "
          f"{sum(1 for c in cases.values() if 'a' in c and 'b' in c)} complete pairs.")

    incomplete = [cid for cid, c in cases.items() if "a" not in c or "b" not in c]
    if incomplete:
        print(f"  [warn] Incomplete pairs: {incomplete}")

    # --- load outcome metadata ---
    print(f"\nLoading outcome metadata ...")
    outcomes = load_metadata(meta_csv)
    if outcomes:
        dist = {}
        for o in outcomes.values():
            dist[o] = dist.get(o, 0) + 1
        print(f"  Outcome distribution: {dist}")
    else:
        print(f"  No metadata loaded -- outcome columns will be blank.")
        print(f"  Re-run with --meta_csv to populate outcomes.")

    # --- optionally copy MIDI files ---
    midi_out = out_dir / "midi"
    if args.copy:
        midi_out.mkdir(parents=True, exist_ok=True)
        print(f"\nCopying MIDI files to {midi_out} ...")
        for cid, c in cases.items():
            for side in ["a", "b"]:
                if side in c:
                    dst = midi_out / c[side].name
                    shutil.copy2(c[side], dst)
        print(f"  Copied {len(list(midi_out.glob('*.mid')))} files.")
    else:
        print(f"\n[info] Skipping file copy (run with --copy to copy MIDI files).")

    # --- write labels.csv ---
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    fieldnames = [
        "case_id", "file_a", "file_b",
        "title_a", "title_b",
        "outcome", "outcome_bin",
    ]

    def file_ref(cid, side):
        """Return relative path string for labels.csv."""
        c = cases.get(cid, {})
        if side not in c:
            return ""
        if args.copy:
            return str(midi_out / c[side].name)
        return str(c[side])   # original path if not copying

    rows = []
    for cid in sorted(cases.keys()):
        outcome = outcomes.get(cid, "")
        rows.append({
            "case_id":     cid,
            "file_a":      file_ref(cid, "a"),
            "file_b":      file_ref(cid, "b"),
            "title_a":     cases[cid].get("title_a", ""),
            "title_b":     cases[cid].get("title_b", ""),
            "outcome":     outcome,
            "outcome_bin": outcome_bin(outcome) if outcome else "",
        })

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {labels_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"MCIC ingest complete.")
    print(f"  labels.csv : {labels_path}")
    print(f"  midi/      : {midi_out} ({'populated' if args.copy else 'not copied'})")
    print(f"\nNext steps:")
    print(f"  1. Export MCIC spreadsheet as CSV and re-run with --meta_csv")
    print(f"     to populate outcome labels.")
    print(f"  2. Run: python scripts/run_validation.py --labels data/mcic/labels.csv")


if __name__ == "__main__":
    main()