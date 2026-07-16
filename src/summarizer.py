"""
BART abstractive summarization module.

This module loads facebook/bart-large-cnn and provides a reusable class
for generating summaries from grouped Amazon Electronics reviews.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class BartSummarizer:
    """Generate abstractive summaries using a pretrained BART model."""

    def __init__(self, model_name: str) -> None:
        """
        Load the tokenizer and pretrained sequence-to-sequence model.

        Args:
            model_name: Hugging Face model identifier.
        """
        print(f"Loading {model_name}...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("BART model loaded.\n")

    def summarize(
        self,
        text: str,
        max_input_tokens: int = 1024,
        max_length: int = 130,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """
        Generate one abstractive summary.

        Args:
            text: Combined review text.
            max_input_tokens: Maximum number of BART input tokens.
            max_length: Maximum summary token length.
            min_length: Minimum summary token length.
            num_beams: Number of beams used during beam search.
            length_penalty: Controls preference for longer summaries.
            no_repeat_ngram_size: Prevents repeated n-grams.

        Returns:
            Generated summary text. Returns an empty string when the
            supplied input contains no usable text.
        """
        cleaned_text = str(text).strip()

        if not cleaned_text:
            return ""

        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=max_input_tokens,
            truncation=True,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        input_token_count = int(encoded["input_ids"].shape[1])

        # Very short inputs do not support a large minimum output length.
        safe_min_length = min(
            min_length,
            max(5, input_token_count // 2),
        )

        safe_max_length = max(
            safe_min_length + 5,
            max_length,
        )

        with torch.inference_mode():
            summary_ids = self.model.generate(
                **encoded,
                max_length=safe_max_length,
                min_length=safe_min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return summary.strip()


def run_smoke_test() -> None:
    """Run a small standalone test of the summarization module."""
    summarizer = BartSummarizer("facebook/bart-large-cnn")

    sample_text = """
    These headphones have excellent sound quality and strong bass.
    The battery lasts more than twenty hours on a single charge.
    Customers report that the headphones remain comfortable during
    long listening sessions. The Bluetooth connection is generally
    stable, although a small number of users reported occasional
    connection delays.
    """

    summary = summarizer.summarize(
        text=sample_text,
        max_input_tokens=1024,
        max_length=60,
        min_length=15,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
    )

    print("Generated Summary:\n")
    print(summary)


if __name__ == "__main__":
    run_smoke_test()