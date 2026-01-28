"""Utility functions for GEPA optimization.

Pure file I/O and import utilities. No dspy dependency.
"""

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def load_import_path(workspace_path: str, dotted_path: str) -> Any:
    """Import an attribute from a dotted path relative to the workspace.

    The last component is the attribute name, everything before is the module.
    e.g., "eval.evaluate.metric" → from eval.evaluate import metric

    Args:
        workspace_path: Path to the cloned repo root (added to sys.path).
        dotted_path: Dotted import path (e.g., "src.module.ClassName").

    Returns:
        The imported attribute (class, function, etc.).
    """
    ws = str(workspace_path)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    module_path, attr_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def load_dataset_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Load a dataset from a file. Supports .json, .jsonl, and .csv.

    Args:
        file_path: Absolute path to the data file.

    Returns:
        List of dicts, one per example.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with open(file_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON file must contain a list of objects, got {type(data).__name__}")

    if suffix == ".jsonl":
        items = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if suffix == ".csv":
        items = []
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items

    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .json, .jsonl, or .csv")
