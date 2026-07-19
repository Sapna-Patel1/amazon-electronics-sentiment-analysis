# M3 Data Pipeline Gap-Closing Design

Date: 2026-07-19
Status: Approved

## Context

Milestone 3's rubric (see `modue_11_req.md`) and the team's own breakdown call
for three pipeline artifacts that don't exist yet, plus a preprocessing filter
that's missing, plus a README that's fallen out of sync with the repo:

- `utils/helpers.py` — shared text-cleaning/label-mapping logic, to remove
  duplication between `src/preprocess.py` and `src/data_loader.py`.
- `configs/data_config.yaml` — centralized, reproducible pipeline settings
  (sample size, split ratios, min reviews/product, min review length, seed).
- A "drop super short entries" filter in `preprocess.py` (currently only
  duplicates and missing text are dropped).
- README fixes: the repo-structure tree diagram references deleted files
  (`data/preprocessing.ipynb`, `processed_reviews.csv.gz`) and puts the EDA
  notebook in `experiments/` when it actually lives in `notebooks/eda.ipynb`.

Explicitly out of scope: the literature-review content for `docs/` and
Prema's third EDA research question. Those require actual research/analysis
work, not something to synthesize here — this design only closes the
code/infra gaps.

## Component 1: `src/utils/helpers.py`

`src/utils.py` (currently a single file exposing `load_config`) becomes a
package: `src/utils/__init__.py` (re-exports `load_config`, unchanged
behavior) + `src/utils/helpers.py` (new). This keeps the existing import
`from utils import load_config` working unchanged for `train.py`/`evaluate.py`,
while satisfying "helpers.py under utils/" for the functions both pipeline
scripts need.

Rationale for this location over a repo-root `utils/`: `data_loader.py` and
`preprocess.py` run as `python src/data_loader.py`, so only `src/` is on
`sys.path` automatically. A sibling top-level `utils/` wouldn't be importable
without extra path configuration.

`helpers.py` contents (moved, not new logic — extracted from `preprocess.py`
and de-duplicated from `data_loader.py`):

- `RATING_TO_LABEL` (dict) and `get_sentiment(rating)` — moved from `preprocess.py`.
- `clean_text(series: pd.Series) -> pd.Series` — the strip + collapse-whitespace
  logic currently inlined three times in `preprocess.py` (for `review_title`,
  `text`, and `review`).
- `split_data(df, train_size=0.70, val_size=0.15, seed=42)` — the single
  shared implementation. Both `preprocess.py` and `data_loader.py` import this
  instead of each defining their own copy.

`preprocess.py` and `data_loader.py` are updated to import from
`utils.helpers` and their local duplicate definitions are deleted.

## Component 2: `configs/data_config.yaml`

```yaml
sampling:
  metadata_sample_size: 10000
  review_sample_size: 50000
  min_reviews_per_product: 10
  chunksize: 50000

split:
  train_size: 0.70
  val_size: 0.15

preprocessing:
  min_review_length: 10

seed: 42
```

(`test_size` is derived as `1 - train_size - val_size`, matching the existing
`split_data` signature — no separate `test_size` key to avoid a value that
could silently disagree with the other two.)

Wiring:
- `data_loader.py`'s `__main__` block loads this config via `load_config`
  (generalized — already accepts a `path` arg) and passes
  `metadata_sample_size`/`review_sample_size`/`min_reviews_per_product`/
  `chunksize` into `load_metadata`/`load_reviews` instead of relying on the
  functions' hardcoded defaults. The function signatures keep their defaults
  (so the functions remain independently callable/testable) but the
  `__main__` entry point becomes config-driven.
- `load_reviews`'s hardcoded `review_counts > 10` becomes
  `review_counts > min_reviews_per_product` (operator unchanged — still
  "more than N", not "at least N" — to preserve exact current behavior),
  parameterized with a default of 10, fed from config in `__main__`.
- `preprocess.py`'s `preprocess_data()` loads this config at the top and
  passes `train_size`/`val_size`/`seed` into `split_data(...)`, and
  `min_review_length` into the new length filter (Component 3).

## Component 3: Short-entry filter

In `preprocess_data()`, immediately after the existing
`dropna(subset=["text"])` + `drop_duplicates()` steps and after `text` is
cleaned (stripped/whitespace-collapsed), add:

```python
merged_df = merged_df[merged_df["text"].str.len() >= min_review_length]
```

Filtering on the cleaned `text` (review body), not the combined
`review`/`input_text` field — a short body with a long title shouldn't count
as "real" content. Threshold: 10 characters (approved). Print a before/after
row count, consistent with the existing print-based diagnostics style in this
file.

## Component 4: README fixes

- Update the "Repository Structure" tree: add `src/utils/` (`__init__.py`,
  `helpers.py`), `configs/data_config.yaml`, move `eda.ipynb` under
  `notebooks/` (not `experiments/`), remove the two stale entries
  (`data/preprocessing.ipynb`, `data/processed_reviews.csv.gz` — already
  deleted from the repo), and add the files currently missing from the tree
  entirely (`src/model_runner.py`, `src/summarizer.py`,
  `src/summary_evaluation.py`, `configs/bart_config.yaml`).
- Add a short paragraph under "Configuration" documenting
  `data_config.yaml` alongside the existing `bert_config.yaml`/
  `bart_config.yaml` descriptions.
- No change to the M3 status row or the "To be completed" markers for Steps
  4-7 — those depend on work (EDA RQ3, docs/ literature review, BERT
  training) outside this PR's scope, and aren't this PR's call to declare
  done.

## Testing / Verification

No test framework exists in this repo. Verification is manual, run end to
end:

1. `python src/data_loader.py` — confirm it still produces
   `data/raw/{metadata,reviews}_sample.csv` with the config-driven sample
   sizes.
2. `python src/preprocess.py` — confirm `data/processed/*.csv` are produced,
   row counts drop appropriately after the new length filter, and the
   train/val/test split ratios match `data_config.yaml`.
3. Diff `helpers.py`'s `split_data`/`get_sentiment` output against the
   previous inline versions on a small sample to confirm the extraction
   didn't change behavior.
4. Confirm `train.py`/`evaluate.py` still import `load_config` successfully
   after `src/utils.py` becomes a package.
