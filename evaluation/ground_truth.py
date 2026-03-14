"""
Ground Truth Module
===================
Loads per-form ground truth files and compares extracted values against them.

Ground truth file format  (one JSON file per completed form):
------------------------------------------------------------
{
  "form_id": "form_001",
  "template_id": "template_001",
  "fields": {
    "name":        "Jane Smith",
    "dob":         "12/03/1995",
    "matric_no":   "20/52RA069",
    "consent":     true,
    "gender_male": false,
    "comments":    "N/A"
  }
}

Naming convention: ground_truth/<form_id>.json
"""

import json
import re
from pathlib import Path


class GroundTruth:
    """
    Loads and indexes all ground truth files in a directory.
    """

    def __init__(self, ground_truth_dir: str = "ground_truth"):
        self._dir   = Path(ground_truth_dir)
        self._cache: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        for path in self._dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                form_id = data.get("form_id", path.stem)
                self._cache[form_id] = data
            except (json.JSONDecodeError, OSError):
                pass

    def get(self, form_id: str) -> dict | None:
        return self._cache.get(form_id)

    def list_form_ids(self) -> list[str]:
        return list(self._cache.keys())


# ── Field-level comparison ─────────────────────────────────────────────────────

def compare_field(
    extracted_value,
    ground_truth_value,
    field_type: str,
) -> dict:
    """
    Compare one extracted field value against its ground truth.

    Returns
    -------
    dict with keys:
      correct          : bool   — primary correctness flag (normalised match)
      exact_match      : bool   — character-perfect match
      normalised_match : bool   — case-insensitive, whitespace-stripped
      similarity       : float  — character-level similarity [0, 1]
    """
    if field_type == "checkbox":
        ext_bool = bool(extracted_value)
        gt_bool  = bool(ground_truth_value)
        match    = ext_bool == gt_bool
        return {
            "correct":          match,
            "exact_match":      match,
            "normalised_match": match,
            "similarity":       1.0 if match else 0.0,
        }

    ext_str = _normalise(str(extracted_value)  if extracted_value  is not None else "")
    gt_str  = _normalise(str(ground_truth_value) if ground_truth_value is not None else "")

    exact      = (str(extracted_value).strip() == str(ground_truth_value).strip())
    normalised = (ext_str == gt_str)
    sim        = _char_similarity(ext_str, gt_str)

    return {
        "correct":          normalised,
        "exact_match":      exact,
        "normalised_match": normalised,
        "similarity":       round(sim, 4),
    }


def _normalise(text: str) -> str:
    """Lowercase, strip extra whitespace, remove common noise characters."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:'\"\-–—]", "", text)
    return text.strip()


def _char_similarity(a: str, b: str) -> float:
    """Simple character-level Jaccard-like similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Use longest-common-subsequence length / max length
    lcs = _lcs_length(a, b)
    return lcs / max(len(a), len(b))


def _lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence (DP, capped at 200 chars)."""
    a, b = a[:200], b[:200]
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]
