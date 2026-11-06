"""
check_smp.py
------------
Diagnose SMP validation failures.
Run from repo root:
    python scripts/check_smp.py
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("=" * 55)
print("SMP Diagnostic")
print("=" * 55)

# 1. Check MIDI files
midi_files = sorted(glob.glob("data/smp/midi/*.mid"))
print(f"\n1. MIDI files in data/smp/midi/: {len(midi_files)}")
for f in midi_files[:6]:
    size = os.path.getsize(f)
    print(f"   {os.path.basename(f):50s}  {size:>6} bytes")

# 2. Check labels.csv
print(f"\n2. labels.csv contents:")
import csv
labels = list(csv.DictReader(open("data/smp/labels.csv", encoding="utf-8")))
print(f"   Total rows: {len(labels)}")
for row in labels[:4]:
    ok = row.get("ok","?")
    ob = row.get("outcome_bin","?")
    fo = Path(row.get("file_ori",""))
    fc = Path(row.get("file_comp",""))
    print(f"   seg={row['seg_id']:<20} ok={ok}  ob={ob}  "
          f"ori_exists={fo.exists()}  comp_exists={fc.exists()}")

# 3. Tokenise first available MIDI pair
print(f"\n3. Tokenisation test:")
try:
    from music_bias_analyzer.tokenizer import file_to_tokens

    usable = [r for r in labels
              if r.get("ok") in ("True","1","true")
              and Path(r["file_ori"]).exists()
              and Path(r["file_comp"]).exists()]
    print(f"   Usable rows (ok + files exist): {len(usable)}")

    if usable:
        row = usable[0]
        ta = file_to_tokens(row["file_ori"])
        tb = file_to_tokens(row["file_comp"])
        print(f"   file_ori  -> {len(ta)} tokens  first5={ta[:5]}")
        print(f"   file_comp -> {len(tb)} tokens  first5={tb[:5]}")

        # 4. Score them
        if ta and tb:
            print(f"\n4. Scoring test:")
            from music_bias_analyzer.lcs import lcs_density_score
            from music_bias_analyzer.ngram import jaccard_overlap
            d = lcs_density_score(ta, tb)
            j = jaccard_overlap(ta, tb, n=5)
            print(f"   lcs_density_score: {d}")
            print(f"   jaccard_overlap:   {j}")
        else:
            print(f"\n4. Skipped -- empty token sequences")
    else:
        print("   No usable rows found -- check ok column and file paths")

except Exception as e:
    import traceback
    print(f"   ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*55)
