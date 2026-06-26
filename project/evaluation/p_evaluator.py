"""Evaluator for .p (pickle) files containing serialized evaluation data."""

import pickle
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_p_file(filepath: str) -> dict[str, Any]:
    """Read and evaluate a .p (pickle) file.
    
    Args:
        filepath: Path to the .p file.
    
    Returns:
        dict with keys:
            - 'filepath': the original path
            - 'success': bool
            - 'data': the unpickled data (if successful)
            - 'data_type': string type name of the data
            - 'error': error message if failed
    """
    path = Path(filepath)
    if not path.exists():
        return {
            "filepath": filepath,
            "success": False,
            "data": None,
            "data_type": None,
            "error": f"File not found: {filepath}",
        }
    if path.suffix.lower() not in (".p", ".pickle", ".pkl"):
        logger.warning("File %s does not have a standard pickle extension", filepath)
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
        return {
            "filepath": filepath,
            "success": True,
            "data": data,
            "data_type": type(data).__name__,
            "error": None,
        }
    except pickle.UnpicklingError as e:
        logger.exception("Failed to unpickle %s", filepath)
        return {
            "filepath": filepath,
            "success": False,
            "data": None,
            "data_type": None,
            "error": f"UnpicklingError: {e}",
        }
    except Exception as e:
        logger.exception("Unexpected error reading %s", filepath)
        return {
            "filepath": filepath,
            "success": False,
            "data": None,
            "data_type": None,
            "error": str(e),
        }


def extract_evaluation_results(filepath: str) -> dict[str, Any]:
    """Extract evaluation results from a .p file with metrics if available.
    
    Looks for common structures in pickled data:
      - dict with 'metrics', 'results', 'fields', 'accuracy' keys
      - list of field results
      - nested dict structures
    
    Args:
        filepath: Path to the .p file.
    
    Returns:
        dict with structured evaluation results plus metadata.
    """
    result = evaluate_p_file(filepath)
    if not result["success"]:
        return result

    data = result["data"]
    extracted = {"metrics": {}, "fields": [], "summary": {}}

    if isinstance(data, dict):
        known_keys = {"metrics", "results", "fields", "accuracy", "precision", "recall", "f1"}
        for key in known_keys & data.keys():
            extracted[key] = data[key]
        if "accuracy" in data:
            extracted["summary"]["accuracy"] = data["accuracy"]
        if all(k in data for k in ("correct", "total")):
            c = int(data["correct"])
            t = int(data["total"])
            extracted["summary"]["accuracy"] = c / t if t > 0 else 0.0
            extracted["summary"]["correct"] = c
            extracted["summary"]["total"] = t
        if "field_results" in data:
            extracted["fields"] = data["field_results"]
        if "summary" in data and isinstance(data["summary"], dict):
            extracted["summary"].update(data["summary"])
    elif isinstance(data, (list, tuple)):
        extracted["fields"] = list(data)
        extracted["summary"] = {"total_items": len(data)}
    else:
        extracted["summary"] = {"data_type": type(data).__name__}

    extracted["success"] = True
    extracted["error"] = None
    return {**result, **extracted}


def batch_evaluate(p_files: list[str]) -> list[dict[str, Any]]:
    """Evaluate multiple .p files and return aggregated results."""
    return [extract_evaluation_results(f) for f in p_files]
