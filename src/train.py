"""
Fine-tune BERT for 3-class sentiment classification.

Loads the processed reviews dataset, tokenizes inputs, and fine-tunes
bert-base-uncased using the Hugging Face Trainer API. Hyperparameters
and paths are read from configs/bert_config.yaml.

Usage:
    python src/train.py
"""

import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_processed_data
from utils import load_config
from utils.helpers import SENTIMENT_LABELS


class ReviewDataset(TorchDataset):
    """PyTorch Dataset for tokenized review text and sentiment labels.

    Args:
        texts: List of input strings (review_title + review text).
        labels: List of integer labels (0=negative, 1=neutral, 2=positive).
        tokenizer: Hugging Face tokenizer instance.
        max_length: Maximum token length for truncation.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        """Tokenize all texts upfront; leave variable-length sequences unpadded."""
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False
        )
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single tokenized sample with its label as tensors."""
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer that applies per-class loss weights (RQ2 class-balancing experiment).

    Args:
        class_weights: 1D tensor of per-class weights matching SENTIMENT_LABELS
            order, or None to fall back to plain unweighted cross-entropy.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    """Compute accuracy and F1 scores from model predictions.

    Tracks overall accuracy, macro-F1, weighted-F1, and neutral-class F1
    separately (neutral is the minority class and key indicator of class-
    balance quality per RQ2).

    Args:
        eval_pred: Tuple of (logits, labels) from the Trainer.

    Returns:
        Dictionary with accuracy, f1_macro, f1_weighted, and f1_neutral.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, labels=SENTIMENT_LABELS, average="macro"),
        "f1_weighted": f1_score(labels, preds, labels=SENTIMENT_LABELS, average="weighted"),
        "f1_neutral": f1_score(labels, preds, labels=SENTIMENT_LABELS, average=None)[1],
    }


def main():
    """Run the full BERT fine-tuning pipeline and save the trained model."""
    cfg = load_config()
    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]
    t = cfg["training"]
    seed = t["seed"]

    print(f"Loading train split from {cfg['data']['train_path']}...")
    train_df = load_processed_data(cfg["data"]["train_path"])
    print(f"Loading validation split from {cfg['data']['val_path']}...")
    val_df = load_processed_data(cfg["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = ReviewDataset(
        train_df["input_text"].tolist(), train_df["label"].tolist(), tokenizer, max_length
    )
    val_ds = ReviewDataset(
        val_df["input_text"].tolist(), val_df["label"].tolist(), tokenizer, max_length
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=cfg["model"]["num_labels"]
    )

    use_class_weights = t.get("use_class_weights", False)
    class_weights = None
    if use_class_weights:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(SENTIMENT_LABELS),
            y=train_df["label"].values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"Class weights (RQ2, balanced): {dict(zip(SENTIMENT_LABELS, weights))}")

    output_dir = cfg["outputs"]["model_dir"] + ("_weighted" if use_class_weights else "_baseline")
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        num_train_epochs=t["num_epochs"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=seed,
        report_to="none",
        # Mixed-precision training roughly halves step time on CUDA GPUs
        # (e.g. Colab's T4). Not supported the same way on MPS/CPU, so
        # only enable it when CUDA is actually available.
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
