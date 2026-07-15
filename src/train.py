"""Fine-tune BERT for 3-class sentiment classification."""

import yaml
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_processed_data, split_data


def load_config(path: str = "configs/bert_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def tokenize(examples, tokenizer, max_length):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_neutral": f1_score(labels, preds, average=None)[1],
    }


def main():
    cfg = load_config()
    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]
    t = cfg["training"]

    print(f"Loading data from {cfg['data']['processed_path']}...")
    df = load_processed_data(cfg["data"]["processed_path"])
    train_df, val_df, _ = split_data(df, seed=cfg["data"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(examples):
        return tokenize(examples, tokenizer, max_length)

    train_ds = Dataset.from_pandas(train_df[["input_text", "label"]]).map(_tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["input_text", "label"]]).map(_tokenize, batched=True)

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
        seed=cfg["data"]["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
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
