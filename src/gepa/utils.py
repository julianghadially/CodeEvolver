"""Utility functions for GEPA optimization."""

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from dspy.primitives import Example


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


def resolve_dataset(
    workspace_path: Path,
    inline_data: list[dict[str, Any]] | None,
    file_path: str | None,
    input_keys: list[str] | None = None,
    required: bool = True,
) -> list[Example] | None:
    """Resolve a dataset from either inline data or a file path.

    Args:
        workspace_path: Root of the cloned repo.
        inline_data: Dataset provided inline in the API request.
        file_path: Path to a data file in the repo (relative).
        input_keys: Fields to mark as inputs via with_inputs().
        required: If True, raises when neither source is provided.

    Returns:
        List of dspy.Example objects, or None if not required and not provided.
    """
    if inline_data is not None:
        raw = inline_data
    elif file_path is not None:
        raw = load_dataset_from_file(workspace_path / file_path)
    elif required:
        raise ValueError("No dataset: provide inline data or a file path")
    else:
        return None

    examples = []
    for item in raw:
        ex = Example(**item)
        if input_keys:
            ex = ex.with_inputs(*input_keys)
        examples.append(ex)
    return examples
