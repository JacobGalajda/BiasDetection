"""
ingest_smp.py
-------------
Download SMP dataset audio from YouTube, extract annotated similar
segments, transcribe to MIDI via librosa pitch tracking, and generate
labels.csv for run_smp_validation.py.

Pipeline per pair:
  1. yt-dlp  -> download full audio as .wav (mono, 22050 Hz)
  2. librosa -> extract 8-second segment windows at annotated timestamps
  3. librosa.pyin -> monophonic pitch tracking on each segment
  4. pretty_midi -> write note sequence as .mid
  5. labels.csv  -> pair_id, seg_id, file_ori, file_comp, relation

Relation labels:
  plag        -> outcome_bin = 1  (confirmed plagiarism)
  remake      -> outcome_bin = 1  (cover -- melodic similarity confirmed)
  plag_doubt  -> outcome_bin = -1 (excluded, ambiguous)

Usage
-----
    python scripts/ingest_smp.py \
        --csv      data/smp/raw/smp_dataset.csv \
        --out_dir  data/smp \
        [--segment_dur 8.0] \
        [--workers 2] \
        [--limit 5]          # for testing: only download first N pairs

Output
------
    data/smp/audio/          -- full downloaded .wav files
    data/smp/segments/       -- extracted 8s segment .wav files
    data/smp/midi/           -- transcribed .mid files per segment
    data/smp/labels.csv      -- one row per segment pair
"""

import argparse
import ast
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(url: str, out_path: Path, sr: int = 22050) -> bool:
    """Download YouTube audio as mono WAV using yt-dlp + ffmpeg."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True  # already downloaded

    tmp = out_path.with_suffix(".%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"ffmpeg:-ac 1 -ar {sr}",
        "--output", str(tmp),
        "--no-playlist",
        "--quiet",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        # yt-dlp writes to out_path.wav
        wav_candidate = out_path.parent / (out_path.stem + ".wav")
        if wav_candidate.exists():
            wav_candidate.rename(out_path)
            return True
        # Check if file appeared at expected path
        if out_path.exists():
            return True
        return False
    except Exception as e:
        print(f"    [download error] {e}")
        return False


# ---------------------------------------------------------------------------
# Vocal stem separation via Demucs
# ---------------------------------------------------------------------------

def separate_vocals(audio_path: Path, stems_dir: Path) -> Path:
    """
    Run Demucs via uvx to separate vocals from mixed audio.
    Returns path to vocals.wav stem, or original path if separation fails.
    """
    stem_name = audio_path.stem
    vocals_path = stems_dir / "htdemucs" / stem_name / "vocals.wav"

    if vocals_path.exists() and vocals_path.stat().st_size > 1000:
        return vocals_path  # already separated

    print(f"    Separating vocals for {stem_name} ...")
    cmd = [
        "uvx", "--with", "soundfile",
        "demucs",
        "--two-stems", "vocals",
        "--out", str(stems_dir),
        "--quiet",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if vocals_path.exists():
            return vocals_path
        print(f"    [demucs] vocals not found after separation")
        return audio_path  # fallback to mix
    except Exception as e:
        print(f"    [demucs error] {e} -- falling back to mix")
        return audio_path


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segment(audio: np.ndarray, sr: int,
                    start_s: float, dur_s: float = 8.0) -> np.ndarray:
    """Extract a fixed-duration segment from audio array."""
    start_sample = int(start_s * sr)
    end_sample   = int((start_s + dur_s) * sr)
    end_sample   = min(end_sample, len(audio))
    if start_sample >= len(audio):
        return np.zeros(int(dur_s * sr))
    return audio[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Pitch tracking -> MIDI
# ---------------------------------------------------------------------------

def segment_to_midi(segment: np.ndarray, sr: int,
                    out_path: Path,
                    fmin: float = 150.0,
                    fmax: float = 2000.0) -> bool:
    """
    Track monophonic pitch in segment using librosa.pyin,
    convert to MIDI note sequence, save as .mid.
    """
    if out_path.exists():
        return True

    try:
        # Separate harmonic content to reduce bass/percussion interference
        segment_harmonic = librosa.effects.harmonic(segment, margin=3.0)

        # pyin: probabilistic YIN -- best for melodic pitch tracking
        # fmin=200Hz focuses on vocal/melody range (above bass)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment_harmonic,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=4096,   # larger frame for better low-freq resolution
            hop_length=256,      # smaller hop for more tokens
        )

        # Convert f0 to MIDI note sequence
        hop_s = 512 / sr  # seconds per frame
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)  # acoustic grand piano

        note_on_time  = None
        note_on_pitch = None

        for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
            t = i * hop_s
            if voiced and freq is not None and not np.isnan(freq):
                pitch = int(round(librosa.hz_to_midi(freq)))
                pitch = max(0, min(127, pitch))

                if note_on_pitch is None:
                    # Start new note
                    note_on_time  = t
                    note_on_pitch = pitch
                elif abs(pitch - note_on_pitch) > 1:
                    # Pitch changed -- close current note, start new
                    if note_on_time is not None and t > note_on_time:
                        note = pretty_midi.Note(
                            velocity=80,
                            pitch=note_on_pitch,
                            start=note_on_time,
                            end=t,
                        )
                        inst.notes.append(note)
                    note_on_time  = t
                    note_on_pitch = pitch
            else:
                # Unvoiced -- close any open note
                if note_on_pitch is not None and note_on_time is not None and t > note_on_time:
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=note_on_pitch,
                        start=note_on_time,
                        end=t,
                    )
                    inst.notes.append(note)
                note_on_pitch = None
                note_on_time  = None

        if inst.notes:
            midi.instruments.append(inst)
            midi.write(str(out_path))
            return True
        else:
            return False  # no pitched content detected

    except Exception as e:
        print(f"    [transcribe error] {e}")
        return False


# ---------------------------------------------------------------------------
# Process one pair
# ---------------------------------------------------------------------------

def process_pair(args):
    (row, audio_dir, stems_dir, seg_dir, midi_dir, seg_dur, sr) = args

    pair_id  = row["pair_number"]
    relation = row["relation"]
    ori_url  = row["ori_link"]
    comp_url = row["comp_link"]
    ori_title = row["ori_title"][:40].replace("/", "_").replace(" ", "_")
    comp_title = row["comp_title"][:40].replace("/", "_").replace(" ", "_")

    try:
        ori_times  = ast.literal_eval(row["ori_times"])
        comp_times = ast.literal_eval(row["comp_times"])
    except Exception:
        return {"pair_id": pair_id, "error": "bad timestamps"}

    # Match segments: use min length of both lists
    n_segs = min(len(ori_times), len(comp_times))
    if n_segs == 0:
        return {"pair_id": pair_id, "error": "no segments"}

    # Download audio
    ori_wav  = audio_dir / f"{pair_id}_{ori_title}.wav"
    comp_wav = audio_dir / f"{pair_id}_{comp_title}.wav"

    print(f"  Pair {pair_id} ({relation}): downloading...")
    ok_ori  = download_audio(ori_url,  ori_wav,  sr)
    ok_comp = download_audio(comp_url, comp_wav, sr)

    if not ok_ori or not ok_comp:
        return {"pair_id": pair_id,
                "error": f"download failed: ori={ok_ori} comp={ok_comp}"}

    # Separate vocals
    ori_vocal  = separate_vocals(ori_wav,  stems_dir)
    comp_vocal = separate_vocals(comp_wav, stems_dir)

    # Load vocals stem (or mix fallback)
    try:
        ori_audio,  _ = librosa.load(str(ori_vocal),  sr=sr, mono=True)
        comp_audio, _ = librosa.load(str(comp_vocal), sr=sr, mono=True)
    except Exception as e:
        return {"pair_id": pair_id, "error": f"load: {e}"}

    # Process each segment pair
    seg_results = []
    for seg_idx in range(n_segs):
        ot = ori_times[seg_idx]
        ct = comp_times[seg_idx]

        seg_id = f"{pair_id}_seg{seg_idx:02d}"

        # Extract segments
        ori_seg  = extract_segment(ori_audio,  sr, ot, seg_dur)
        comp_seg = extract_segment(comp_audio, sr, ct, seg_dur)

        # Transcribe to MIDI
        ori_mid  = midi_dir / f"{seg_id}_ori.mid"
        comp_mid = midi_dir / f"{seg_id}_comp.mid"

        ok_ori_mid  = segment_to_midi(ori_seg,  sr, ori_mid)
        ok_comp_mid = segment_to_midi(comp_seg, sr, comp_mid)

        seg_results.append({
            "pair_id":     pair_id,
            "seg_id":      seg_id,
            "relation":    relation,
            "outcome_bin": 1 if relation in ("plag", "remake") else -1,
            "ori_start_s": ot,
            "comp_start_s": ct,
            "file_ori":    str(ori_mid)  if ok_ori_mid  else "",
            "file_comp":   str(comp_mid) if ok_comp_mid else "",
            "ok":          ok_ori_mid and ok_comp_mid,
        })

    print(f"  Pair {pair_id}: {sum(1 for s in seg_results if s['ok'])}"
          f"/{n_segs} segments transcribed")
    return {"pair_id": pair_id, "segments": seg_results, "error": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",         default="data/smp/raw/smp_dataset.csv")
    ap.add_argument("--out_dir",     default="data/smp")
    ap.add_argument("--segment_dur", type=float, default=15.0,
                    help="Segment duration in seconds (default: 8.0)")
    ap.add_argument("--sr",          type=int, default=22050,
                    help="Audio sample rate")
    ap.add_argument("--workers",     type=int, default=2,
                    help="Parallel download workers (keep low to avoid yt-dlp rate limiting)")
    ap.add_argument("--limit",       type=int, default=0,
                    help="Only process first N pairs (0 = all)")
    args = ap.parse_args()

    out_dir   = Path(args.out_dir)
    audio_dir = out_dir / "audio";    audio_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = out_dir / "stems";    stems_dir.mkdir(parents=True, exist_ok=True)
    seg_dir   = out_dir / "segments"; seg_dir.mkdir(parents=True, exist_ok=True)
    midi_dir  = out_dir / "midi";     midi_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV -- deduplicate by pair_number (multiple rows per pair = multiple segment groups)
    print(f"Loading {args.csv} ...")
    rows = []
    seen_pairs = set()
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["pair_number"]
            # Each unique (pair_number, ori_times) is a distinct segment group
            key = (pid, row["ori_times"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                rows.append(row)

    if args.limit:
        rows = rows[:args.limit]

    # Count by relation
    from collections import Counter
    rel_counts = Counter(r["relation"] for r in rows)
    print(f"  Segment groups: {len(rows)}  {dict(rel_counts)}")
    print(f"  Usable (plag+remake): "
          f"{sum(1 for r in rows if r['relation'] in ('plag','remake'))}")
    print(f"  Excluded (plag_doubt): "
          f"{sum(1 for r in rows if r['relation'] == 'plag_doubt')}")

    # Build worker args
    worker_args = [
        (row, audio_dir, stems_dir, seg_dir, midi_dir, args.segment_dur, args.sr)
        for row in rows
    ]

    print(f"\nProcessing {len(worker_args)} segment groups "
          f"with {args.workers} worker(s) ...")

    # Use workers=1 to avoid yt-dlp rate limiting
    if args.workers == 1:
        pair_results = [process_pair(a) for a in worker_args]
    else:
        with mp.Pool(args.workers) as pool:
            pair_results = pool.map(process_pair, worker_args)

    # Flatten segments into labels.csv
    labels_path = out_dir / "labels.csv"
    fieldnames  = [
        "pair_id", "seg_id", "relation", "outcome_bin",
        "ori_start_s", "comp_start_s",
        "file_ori", "file_comp", "ok",
    ]

    all_segments = []
    errors       = []
    for pr in pair_results:
        if pr.get("error"):
            errors.append(pr)
        else:
            all_segments.extend(pr.get("segments", []))

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_segments)

    ok_segs  = sum(1 for s in all_segments if s["ok"])
    plag_segs = sum(1 for s in all_segments if s["outcome_bin"] == 1 and s["ok"])
    excl_segs = sum(1 for s in all_segments if s["outcome_bin"] == -1)

    print(f"\n{'='*50}")
    print(f"SMP ingest complete")
    print(f"  Total segment pairs:  {len(all_segments)}")
    print(f"  Successfully transcribed: {ok_segs}")
    print(f"  Usable (plag+remake): {plag_segs}")
    print(f"  Excluded (plag_doubt): {excl_segs}")
    print(f"  Errors: {len(errors)}")
    print(f"  labels.csv: {labels_path}")
    if errors:
        for e in errors[:3]:
            print(f"    Pair {e['pair_id']}: {e['error']}")
    print(f"\nNext: python scripts/run_smp_validation.py --labels {labels_path}")


if __name__ == "__main__":
    main()