"""Shared utilities for the sentiment analysis pipeline."""

import yaml


def load_config(path: str = "configs/bert_config.yaml") -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    with open(path) as f:
        return yaml.safe_load(f)
