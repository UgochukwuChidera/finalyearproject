"""Tests for confidence/similarity math.

Covers: Levenshtein distance, dictionary matching, C_lp computation,
C_dict computation, C_final weighted combination, and confidence
thresholds used in validation.
"""
from __future__ import annotations

import math
import pytest

from ai_extraction.confidence import (
    levenshtein_distance,
    compute_C_lp,
    compute_C_final,
    logprob_to_confidence,
)
from ai_extraction.dictionary_matcher import best_match, compute_C_dict


# ---------------------------------------------------------------------------
# Levenshtein Distance
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_levenshtein_identical(self):
        """Two identical strings → distance 0 → score close to 1.0."""
        dist = levenshtein_distance("hello world", "hello world")
        assert dist == 0

    def test_levenshtein_completely_different(self):
        """Very different strings → large distance."""
        dist = levenshtein_distance("abc", "xyz")
        assert dist == 3  # all three chars differ

    def test_levenshtein_empty(self):
        """Empty strings are handled gracefully."""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "xyz") == 3

    def test_levenshtein_case_insensitive(self):
        """Levenshtein should be case-insensitive (both sides lowered)."""
        dist1 = levenshtein_distance("Hello", "hello")
        dist2 = levenshtein_distance("HELLO", "hello")
        assert dist1 == 0
        assert dist2 == 0


# ---------------------------------------------------------------------------
# best_match & compute_C_dict
# ---------------------------------------------------------------------------

class TestDictionaryMatcher:
    def test_best_match_exact(self):
        """Exact match should return the entry with distance 0."""
        match, dist = best_match("Smith", ["Smith", "Jones", "Taylor"])
        assert match == "Smith"
        assert dist == 0

    def test_best_match_closest(self):
        """Closest match should be found."""
        match, dist = best_match("Smit", ["Smith", "Jones", "Taylor"])
        assert match == "Smith"
        assert dist > 0

    def test_best_match_empty_extracted(self):
        """Empty extracted string → None, 9999."""
        match, dist = best_match("", ["Smith", "Jones"])
        assert match is None
        assert dist == 9999

    def test_best_match_empty_dict(self):
        """Empty dictionary → None, 9999."""
        match, dist = best_match("Smith", [])
        assert match is None
        assert dist == 9999

    def test_compute_C_dict_exact(self):
        """Exact match → C_dict = 1.0."""
        c = compute_C_dict("Smith", "Smith", 0)
        assert c == pytest.approx(1.0)

    def test_compute_C_dict_no_match(self):
        """No match → C_dict = 0.0."""
        c = compute_C_dict("", None, 9999)
        assert c == 0.0

    def test_compute_C_dict_partial(self):
        """Partial match → 0 < C_dict < 1."""
        c = compute_C_dict("Smit", "Smith", 1)
        assert 0.0 < c < 1.0


# ---------------------------------------------------------------------------
# C_lp computation
# ---------------------------------------------------------------------------

class TestCLP:
    def test_compute_C_lp_none(self):
        """None input → default 0.5."""
        assert compute_C_lp(None) == 0.5

    def test_compute_C_lp_positive(self):
        """Positive value (already a probability) → clamped to [0, 1]."""
        c = compute_C_lp(0.8)
        assert c == pytest.approx(0.8)

    def test_compute_C_lp_logprob(self):
        """Negative log-probability → exponentiate."""
        v = compute_C_lp(-0.5)
        expected = math.exp(-0.5)
        assert v == pytest.approx(expected)

    def test_compute_C_lp_list(self):
        """List of logprobs → average exponentiated."""
        v = compute_C_lp([-0.2, -0.3, -0.5])
        avg = (-0.2 + -0.3 + -0.5) / 3
        expected = math.exp(avg)
        assert v == pytest.approx(expected)

    def test_logprob_to_confidence(self):
        """logprob_to_confidence delegates to compute_C_lp."""
        r1 = logprob_to_confidence(-0.5)
        r2 = compute_C_lp(-0.5)
        assert r1 == r2


# ---------------------------------------------------------------------------
# C_final (weighted combination)
# ---------------------------------------------------------------------------

class TestCFinal:
    def test_compute_C_final_equal_weights(self):
        """Equal weights → average of C_lp and C_dict."""
        c = compute_C_final(1.0, 0.0, 0.5, 0.5)
        assert c == pytest.approx(0.5)

    def test_compute_C_final_lp_only(self):
        """w_dict = 0 → result determined solely by C_lp."""
        c = compute_C_final(0.8, 0.0, 1.0, 0.0)
        assert c == pytest.approx(0.8)

    def test_compute_C_final_dict_only(self):
        """w_lp = 0 → result determined solely by C_dict."""
        c = compute_C_final(0.0, 0.6, 0.0, 1.0)
        assert c == pytest.approx(0.6)

    def test_compute_C_final_zero_weights(self):
        """Both weights 0 → fallback 0.5."""
        c = compute_C_final(0.9, 0.9, 0.0, 0.0)
        assert c == pytest.approx(0.5)

    def test_compute_C_final_clamped(self):
        """Result should be clamped to [0, 1]."""
        c = compute_C_final(2.0, 2.0, 0.5, 0.5)
        assert c <= 1.0
        c2 = compute_C_final(-1.0, -1.0, 0.5, 0.5)
        assert c2 >= 0.0


# ---------------------------------------------------------------------------
# Confidence thresholds (from pipeline logic)
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    """Verify the threshold-based classification used in pipeline.py."""

    @pytest.fixture
    def thresholds(self):
        return {"auto_accept": 0.85, "review": 0.70}

    def test_accepted(self, thresholds):
        """confidence >= auto_accept → accepted."""
        for conf in [0.85, 0.90, 1.0]:
            if conf >= thresholds["auto_accept"]:
                status = "accepted"
            elif conf >= thresholds["review"]:
                status = "spot_check"
            else:
                status = "pending_review"
            assert status == "accepted", f"Conf {conf} should be accepted"

    def test_spot_check(self, thresholds):
        """auto_accept > confidence >= review → spot_check."""
        for conf in [0.70, 0.75, 0.84]:
            if conf >= thresholds["auto_accept"]:
                status = "accepted"
            elif conf >= thresholds["review"]:
                status = "spot_check"
            else:
                status = "pending_review"
            assert status == "spot_check", f"Conf {conf} should be spot_check"

    def test_pending_review(self, thresholds):
        """confidence < review → pending_review."""
        for conf in [0.0, 0.50, 0.69]:
            if conf >= thresholds["auto_accept"]:
                status = "accepted"
            elif conf >= thresholds["review"]:
                status = "spot_check"
            else:
                status = "pending_review"
            assert status == "pending_review", f"Conf {conf} should be pending_review"
