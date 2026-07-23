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
- `bert_sample_predictions_baseline.csv` / `bert_sample_predictions_weighted.csv` —
  10 randomly sampled test-set rows per variant, each with the review text,
  actual sentiment, and predicted sentiment, for manual spot-checking.

**Status: done.** Both variants trained and evaluated on an NVIDIA V100 GPU.
Re-run `experiments/bert_class_balancing.ipynb` to reproduce (also regenerates
the full comparison table and confusion matrices in the README).

The full RQ1/RQ2 numeric comparison (accuracy/F1 deltas, confusion matrices,
and trade-off interpretation) lives in the README's
[Preliminary BERT Results](../README.md#preliminary-bert-results) section
rather than duplicated here. For how class weighting is actually computed
(inverse class frequency via scikit-learn's `compute_class_weight`), see the
README's [Step 4](../README.md#step-4-train-the-bert-sentiment-classifier).

## BART review summarization (`src/model_runner.py`)

- `summary_samples.csv` — generated abstractive summaries for each evaluated
  product. Each record includes the product identifier, product title,
  summarization strategy, sentiment group, source review text, generated
  summary, and generation status.
- `summary_evaluation.csv` — per-summary evaluation metrics including source
  and summary lengths, compression ratio, lexical coverage, novelty ratio,
  repetition ratio, sentiment alignment, ROUGE source-coverage metrics, and
  blank fields for manual qualitative evaluation (relevance, coherence,
  conciseness, and pros/cons coverage).
- `strategy_comparison.csv` — aggregated metrics comparing the combined-review
  and sentiment-separated summarization strategies.

**Status: completed.** The BART summarization pipeline executes end-to-end by
running `python src/model_runner.py`. The pipeline automatically loads the
processed review dataset, selects representative products, groups reviews by
product and sentiment, generates abstractive summaries using
`facebook/bart-large-cnn`, evaluates the generated summaries, and saves all
outputs to the `outputs/` directory without requiring manual intervention.

The latest successful execution processed **48,252 reviews**, selected
**20 representative products**, generated **80 summaries**
(20 combined-review summaries and 60 sentiment-separated summaries),
and completed with **0 failed generations**.

### Analysis

The current implementation evaluates two summarization strategies:

- **Combined reviews** — summarizes all selected reviews for a product into a
  single overview intended to capture the overall customer opinion.
- **Sentiment-separated reviews** — generates independent summaries for
  positive, neutral, and negative review groups, allowing strengths,
  weaknesses, and mixed customer feedback to be analyzed separately.

The latest evaluation indicates that the sentiment-separated strategy produces
more focused summaries while preserving the intended sentiment categories.
The combined strategy provides broader product overviews by integrating all
available customer opinions into a single summary.

The full aggregate metrics table (compression ratio, lexical coverage,
novelty/repetition ratios, sentiment alignment, ROUGE source-coverage) and
the RQ3 discussion live in the README's
[Preliminary BART Results](../README.md#preliminary-bart-results) section
rather than duplicated here, so there's one source of truth as the pipeline
is re-run — see `outputs/strategy_comparison.csv` for the underlying numbers.

The manual qualitative evaluation fields (relevance, coherence,
conciseness, and pros/cons coverage) have been scored on a representative
subset — see the README's
[Manual Qualitative Evaluation](../README.md#manual-qualitative-evaluation)
section.
