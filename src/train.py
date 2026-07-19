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

from data_loader import load_processed_data
from utils import load_config

SENTIMENT_LABELS = [0, 1, 2]


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
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
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

    output_dir = cfg["outputs"]["model_dir"]
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
    )

    trainer = Trainer(
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
