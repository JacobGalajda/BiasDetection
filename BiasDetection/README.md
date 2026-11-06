# music_bias_analyzer

NLP-style symbolic music similarity analysis for detecting memorisation
and style bias in note-generation LLMs (NotaGen, Suno, Udio, etc.).

---

## Installation

```bash
pip install music21 scikit-learn numpy scipy matplotlib
```

---

## Quick Start

```python
from music_bias_analyzer import AnalysisPipeline, CorpusGroup

pipe = AnalysisPipeline()
pipe.add_corpus(CorpusGroup.LLM_OUTPUT,      "path/to/notagen_outputs/")
pipe.add_corpus(CorpusGroup.TRAIN_CORPUS,    "path/to/fine_tune_corpus/")
pipe.add_corpus(CorpusGroup.DUPED,           "path/to/suspected_dupes/")
pipe.add_corpus(CorpusGroup.HUMAN_BASELINE,  "path/to/maestro_subset/")

report = pipe.run()
print(report.summary())
report.save("results/report.json")
```

Run the self-contained demo (no music files needed):
```bash
python music_bias_analyzer/demo.py
```

---

## Architecture

```
music_bias_analyzer/
|-- tokenizer.py      # MIDI/ABC -> token sequence ("C4_quarter", "REST_eighth" ...)
|-- ngram.py          # N-gram Jaccard + BLEU precision  (verbatim copy)
|-- lcs.py            # LCS similarity + sliding-window  (paraphrase copy)
|-- embeddings.py     # TF-IDF + pitch-class histogram   (style collapse)
|-- pipeline.py       # Full pairwise orchestration + AnalysisReport
|-- visualise.py      # Matplotlib figures (PCA, heatmap, LCS profile ...)
|-- demo.py           # Synthetic smoke-test / usage example
\-- requirements.txt
```

---

## Metrics Explained

| Metric | Detects | Flag threshold |
|---|---|---|
| Jaccard overlap (n=5) | Shared melodic n-grams | > 0.15 |
| BLEU precision (n=5) | LLM output echoes training | > 0.20 |
| LCS (strict) | Verbatim shared subsequence | > 0.35 |
| LCS (pitch-only) | Rhythmic paraphrase | > 0.40 |
| TF-IDF cosine dist | Style collapse (low = bad) | < 0.25 |
| Histogram dist | Same tonal fingerprint | < 0.10 |

### Why multiple metrics?

Each metric has a different sensitivity/specificity trade-off:

- **N-gram overlap** is the strictest -- it requires the *exact same token* to appear
  at the same relative ordering. Transposed or rhythmically varied copies evade it.
  Good for: proving verbatim memorisation.

- **LCS** allows gaps between matches so it catches paraphrasing where a melody
  is preserved but decorated with passing tones or ornaments.
  Use with `ignore_duration=True` to also catch rhythmic paraphrasing.

- **Embedding distance** captures global style rather than specific passages.
  A low intra-group distance for LLM outputs (style collapse) means the model
  generates music that always sounds the same regardless of prompt.

---

## Experiment Groups

| Group | Description |
|---|---|
| `LLM_OUTPUT` | Generated pieces under evaluation (e.g. NotaGen ABC files) |
| `TRAIN_CORPUS` | Fine-tuning / pre-training data (suspected source) |
| `DUPED` | Manually documented suspected duplicates from prior experiments |
| `HUMAN_BASELINE` | Held-out human-composed reference (MAESTRO subset recommended) |

Cross-group pairs (LLM_OUTPUT <-> TRAIN_CORPUS) with high scores are the
primary signal for memorisation bias.

---

## Evaluation Protocol (for publication)

### Comparison design
1. Compute all three metrics for **LLM  x  Training** pairs
2. Compute same metrics for **Human  x  Training** pairs (null baseline)
3. If LLM scores > Human scores by a meaningful margin -> evidence of bias

### Thresholds
Thresholds in `PipelineConfig` are conservative starting points.
Calibrate them using the Human  x  Training distribution as your null:
- Set flag threshold = u_null + 2sigma_null (two-sigma rule)

### Negative results
If no statistically significant difference is found between LLM and
human baselines:
- The result *bounds* the degree of memorisation
- Frame as: "memorisation, if present, is at or below human-level
  musical similarity which arises naturally from shared tonal conventions"
- Still publishable as a methodological contribution and safety baseline

### Limitations to address
- "Metrics too simple": frame as diagnostic baseline; simplicity aids adoption
  and replication by future researchers
- "Only one model": closed-source models (Suno/Udio) prevent direct replication;
  call for open model audits; NotaGen provides a reproducible test case
- "No clear signal": negative result bounds memorisation and establishes
  the protocol for future work

---

## Visualisations

```python
from music_bias_analyzer.visualise import (
    plot_pca, plot_heatmap, plot_style_collapse,
    plot_lcs_profile, save_all_figures
)

# Save all standard figures from a completed report
save_all_figures(report, output_dir="results/figures/")

# Sliding-window LCS for a specific suspect pair
from music_bias_analyzer.lcs import sliding_window_lcs
profile = sliding_window_lcs(query_tokens, reference_tokens, window=50)
fig = plot_lcs_profile(profile, "notagen_001", "bach_bwv772")
fig.savefig("lcs_profile.png", dpi=150)
```

---

## Supported File Formats

music21 supports: `.mid`, `.midi`, `.abc`, `.mxl`, `.xml`, `.musicxml`

For NotaGen ABC output, use `tokens_from_string()` to tokenise in-memory
without saving to disk first.
