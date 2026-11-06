"""
music_bias_analyzer
===================
NLP-style symbolic music similarity analysis for LLM bias detection.

Public API
----------
    from music_bias_analyzer import AnalysisPipeline, CorpusGroup, PipelineConfig

    # Basic run
    pipe = AnalysisPipeline()
    pipe.add_corpus(CorpusGroup.LLM_OUTPUT,     "outputs/")
    pipe.add_corpus(CorpusGroup.TRAIN_CORPUS,   "train/")
    pipe.add_corpus(CorpusGroup.HUMAN_BASELINE, "maestro/")
    report = pipe.run()

    # With Pareto calibration + ensemble voting
    from music_bias_analyzer import ThresholdCalibrator, EnsembleConfig
    report = pipe.run(
        calibrator = ThresholdCalibrator(),
        ensemble   = EnsembleConfig(metrics=["jaccard_n5", "lcs_strict"]),
    )
    print(report.summary())
    report.save("results/report.json")

Lower-level modules can be imported directly:
    from music_bias_analyzer.tokenizer  import file_to_tokens, load_corpus
    from music_bias_analyzer.ngram      import jaccard_overlap, bleu_precision, candidate_pairs
    from music_bias_analyzer.lcs        import lcs_similarity, lcs_rough, lcs_density_score
    from music_bias_analyzer.embeddings import (TFIDFMusicEmbedder,
                                                semitone_interval_histogram,
                                                music_feature_vector_extended)
    from music_bias_analyzer.visualise  import plot_pca, plot_lcs_profile
"""

from .pipeline   import (AnalysisPipeline, CorpusGroup, PipelineConfig,
                          AnalysisReport, CalibratedThresholds,
                          ThresholdCalibrator, EnsembleConfig, EnsembleVoter)
from .tokenizer  import file_to_tokens, tokens_from_string, load_corpus
from .ngram      import (jaccard_overlap, bleu_precision, multi_n_overlap,
                         candidate_pairs)
from .lcs        import (lcs_similarity, lcs_length, lcs_rough, lcs_density_score,
                         sliding_window_lcs, peak_lcs_region)
from .embeddings import (TFIDFMusicEmbedder, pitch_class_histogram,
                         interval_histogram, semitone_interval_histogram,
                         music_feature_vector, music_feature_vector_extended,
                         style_collapse_score)

__version__ = "0.2.0"
__all__ = [
    # Pipeline
    "AnalysisPipeline", "CorpusGroup", "PipelineConfig", "AnalysisReport",
    "CalibratedThresholds", "ThresholdCalibrator", "EnsembleConfig", "EnsembleVoter",
    # Tokenizer
    "file_to_tokens", "tokens_from_string", "load_corpus",
    # N-gram
    "jaccard_overlap", "bleu_precision", "multi_n_overlap", "candidate_pairs",
    # LCS
    "lcs_similarity", "lcs_length", "lcs_rough", "lcs_density_score",
    "sliding_window_lcs", "peak_lcs_region",
    # Embeddings
    "TFIDFMusicEmbedder", "pitch_class_histogram", "interval_histogram",
    "semitone_interval_histogram", "music_feature_vector",
    "music_feature_vector_extended", "style_collapse_score",
]
