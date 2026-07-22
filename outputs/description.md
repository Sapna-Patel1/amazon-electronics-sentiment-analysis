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
**5 representative products**, generated **20 summaries** (5 combined-review
summaries and 15 sentiment-separated summaries), and completed with
**0 failed generations**.

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

Current aggregate metrics include:

|         Metric          | Combined | Sentiment-Separated |
|-------------------------|---------:|--------------------:|
|   Summaries generated   |     5    |         15          |
|  Average source words   |  2037.6  |        677.2        |
|  Average summary words  |   61.80  |        40.73        |
|    Compression ratio    |   0.0312 |       0.0704        |
|     Lexical coverage    |   99.57% |       97.24%        |
|      Novelty ratio      |   0.44%  |        2.76%        |
|     Repetition ratio    |  13.49%  |        7.30%        |
|     Sentiment alignment |    N/A   |       73.33%        |
| ROUGE-1 source coverage |  0.0310  |       0.0695        |
| ROUGE-2 source coverage |  0.0288  |       0.0596        |
| ROUGE-L source coverage |  0.0302  |       0.0672        |

Overall, the current implementation successfully generates concise,
product-level abstractive summaries while maintaining strong lexical grounding
to the source reviews. Separating reviews by sentiment results in shorter,
more focused inputs and provides clearer distinctions between positive,
neutral, and negative customer feedback.

The manual qualitative evaluation fields (relevance, coherence,
conciseness, and pros/cons coverage) remain available for future human
assessment during the final project evaluation.
