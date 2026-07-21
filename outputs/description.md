# Outputs

This directory holds generated results from both model pipelines. Files here
are committed to the repo (not gitignored) so results are visible without
re-running anything.

## BERT sentiment classification (`src/train.py` / `src/evaluate.py`)

- `bert_evaluation_baseline.txt` / `bert_evaluation_baseline.json` —
  accuracy, precision, recall, F1 (macro/weighted/per-class), and confusion
  matrix for the baseline run (`use_class_weights: false`).
- `bert_evaluation_weighted.txt` / `bert_evaluation_weighted.json` —
  the same metrics for the class-weighted run (`use_class_weights: true`),
  used to answer RQ2 (does class balancing improve neutral-class F1/recall?).

**Status: done.** Both variants trained and evaluated on an NVIDIA V100 GPU.
Re-run `experiments/bert_class_balancing.ipynb` to reproduce (also regenerates
the full comparison table and confusion matrices in the README).

### Analysis

- **RQ1 (accuracy):** the baseline model reaches 90.5% overall accuracy, but
  this is misleading on its own — performance is very uneven across classes
  (F1 0.96 positive, 0.85 negative, only 0.43 neutral). Positive reviews
  dominate the test set (5,366 of 7,239 rows), so overall accuracy mostly
  reflects how well the model does on the majority class.
- **RQ2 (class balancing):** weighting the loss by inverse class frequency
  improves neutral-class F1 by +0.029 and recall by +0.060, at essentially no
  cost to overall accuracy (+0.0001) or weighted F1 (+0.003). The gain isn't
  free — it trades a small amount of positive-class recall (-0.009) and
  negative-class precision (-0.006) for the neutral-class improvement.
  Class balancing helps the specific problem it targets, but neutral reviews
  remain the hardest class by a wide margin even after weighting.
- See the README's "Preliminary BERT Results" section for the full
  baseline-vs-weighted comparison table and both confusion matrices.

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

### Analysis

Pending the actual CSVs being committed (see status above). Once
`summary_evaluation.csv`/`strategy_comparison.csv` are in this directory, this
section should cover: how compression ratio/lexical coverage/novelty/
repetition compare between the combined and sentiment-separated strategies,
whether sentiment alignment holds up across all evaluated products (not just
the aggregate 73.33%), and the manual qualitative scores (relevance,
coherence, conciseness, pros/cons coverage) once filled in — this is
Jose's/the BART side's analysis to write, since it needs the real per-summary
data, not just the aggregate numbers currently in the README.
