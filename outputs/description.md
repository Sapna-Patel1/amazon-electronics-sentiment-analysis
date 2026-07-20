# Outputs

This directory holds generated results from both model pipelines. Files here
are committed to the repo (not gitignored) so results are visible without
re-running anything.

## BERT sentiment classification (`src/train.py` / `src/evaluate.py`)

- `evaluation_results_baseline.txt` / `evaluation_results_baseline.json` —
  accuracy, precision, recall, F1 (macro/weighted/per-class), and confusion
  matrix for the baseline run (`use_class_weights: false`).
- `evaluation_results_weighted.txt` / `evaluation_results_weighted.json` —
  the same metrics for the class-weighted run (`use_class_weights: true`),
  used to answer RQ2 (does class balancing improve neutral-class F1/recall?).

**Status: done.** Both variants trained and evaluated on an NVIDIA V100 GPU.
See the README's "Preliminary BERT Results" section for the comparison table,
confusion matrices, and RQ1/RQ2 answers. Re-run
`experiments/bert_class_balancing.ipynb` to reproduce.

## BART review summarization (`src/model_runner.py`)

- `summary_samples.csv` — generated summaries per product/strategy/sentiment
  group, with source review text and generation status.
- `summary_evaluation.csv` — per-summary diagnostics (compression ratio,
  lexical coverage, novelty, repetition, sentiment alignment, ROUGE
  source-coverage) plus blank fields for manual qualitative scoring
  (relevance, coherence, conciseness, pros/cons coverage).
- `strategy_comparison.csv` — aggregated metrics comparing the combined vs.
  sentiment-separated summarization strategies.

**Status: preliminary results exist** (per the README's "Preliminary BART
Results" section — 15 summaries generated across 5 products, 73.33%
sentiment alignment, 97.24% lexical coverage) but the CSV files themselves
aren't committed here yet. Re-run `python src/model_runner.py` to regenerate
and commit them.
