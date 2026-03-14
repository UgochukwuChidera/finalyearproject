"""
Metrics Module
==============
Computes all evaluation metrics defined in Section 3.5 of the project:

  - Field-level accuracy    (primary)
  - Checkbox precision & recall
  - Escalation rate
  - Auto-accept rate
  - Average processing time per form
  - Pre-HITL vs post-HITL accuracy (HITL impact)

All functions operate on lists of field comparison records and are
independent of which pipeline produced the fields.
"""

from collections import defaultdict


# ── Per-form metrics ───────────────────────────────────────────────────────────

def compute_form_metrics(
    comparisons: list[dict],
    hitl_stats:  dict,
    processing_time_s: float,
) -> dict:
    """
    Compute metrics for a single processed form.

    Parameters
    ----------
    comparisons : list of dicts from compare_field(), each also containing
                  field_type and needs_review / corrected from validation
    hitl_stats  : from HITLEscalation.escalation_stats()
    processing_time_s : wall-clock time for extraction stage

    Returns
    -------
    dict of metric values for this form
    """
    total      = len(comparisons)
    if total == 0:
        return _empty_form_metrics()

    correct    = sum(1 for c in comparisons if c["correct"])
    exact      = sum(1 for c in comparisons if c["exact_match"])
    similarity = sum(c["similarity"] for c in comparisons) / total

    # Checkbox sub-metrics
    cb = [c for c in comparisons if c["field_type"] == "checkbox"]
    cb_metrics = _checkbox_metrics(cb)

    # Text sub-metrics
    txt = [c for c in comparisons if c["field_type"] != "checkbox"]
    txt_accuracy = sum(1 for c in txt if c["correct"]) / max(len(txt), 1)

    return {
        "field_accuracy":        round(correct / total, 4),
        "exact_match_rate":      round(exact / total, 4),
        "avg_similarity":        round(similarity, 4),
        "text_field_accuracy":   round(txt_accuracy, 4),
        "checkbox_precision":    cb_metrics["precision"],
        "checkbox_recall":       cb_metrics["recall"],
        "checkbox_f1":           cb_metrics["f1"],
        "total_fields":          total,
        "correct_fields":        correct,
        "escalation_rate":       hitl_stats.get("escalation_rate", 0.0),
        "corrected_count":       hitl_stats.get("corrected_count", 0),
        "processing_time_s":     round(processing_time_s, 4),
    }


def _checkbox_metrics(cb_comparisons: list[dict]) -> dict:
    if not cb_comparisons:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for c in cb_comparisons if c.get("extracted_bool") and c["correct"])
    fp = sum(1 for c in cb_comparisons if c.get("extracted_bool") and not c["correct"])
    fn = sum(1 for c in cb_comparisons if not c.get("extracted_bool") and not c["correct"])

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def _empty_form_metrics() -> dict:
    return {
        "field_accuracy": 0.0, "exact_match_rate": 0.0, "avg_similarity": 0.0,
        "text_field_accuracy": 0.0, "checkbox_precision": 0.0,
        "checkbox_recall": 0.0, "checkbox_f1": 0.0,
        "total_fields": 0, "correct_fields": 0,
        "escalation_rate": 0.0, "corrected_count": 0,
        "processing_time_s": 0.0,
    }


# ── Aggregate metrics across all forms ────────────────────────────────────────

def aggregate_metrics(form_metrics_list: list[dict]) -> dict:
    """
    Average per-form metrics across an entire dataset.
    Also computes standard deviation for key metrics.
    """
    if not form_metrics_list:
        return {}

    numeric_keys = [
        "field_accuracy", "exact_match_rate", "avg_similarity",
        "text_field_accuracy", "checkbox_precision", "checkbox_recall",
        "checkbox_f1", "escalation_rate", "processing_time_s",
    ]

    totals = defaultdict(float)
    squares = defaultdict(float)
    n = len(form_metrics_list)

    total_fields   = sum(m["total_fields"]   for m in form_metrics_list)
    correct_fields = sum(m["correct_fields"] for m in form_metrics_list)

    for m in form_metrics_list:
        for k in numeric_keys:
            v = float(m.get(k, 0.0))
            totals[k]  += v
            squares[k] += v * v

    agg = {
        "n_forms":               n,
        "overall_field_accuracy": round(correct_fields / max(total_fields, 1), 4),
    }

    for k in numeric_keys:
        mean = totals[k] / n
        var  = (squares[k] / n) - (mean ** 2)
        std  = max(var, 0) ** 0.5
        agg[f"mean_{k}"]   = round(mean, 4)
        agg[f"std_{k}"]    = round(std, 4)

    return agg


# ── HITL impact summary ────────────────────────────────────────────────────────

def hitl_impact(
    pre_hitl_metrics:  dict,
    post_hitl_metrics: dict,
) -> dict:
    """
    Compute the accuracy gain attributable to HITL correction.
    """
    pre  = pre_hitl_metrics.get("mean_field_accuracy",  pre_hitl_metrics.get("field_accuracy",  0.0))
    post = post_hitl_metrics.get("mean_field_accuracy", post_hitl_metrics.get("field_accuracy", 0.0))
    gain = post - pre

    return {
        "pre_hitl_accuracy":  round(pre,  4),
        "post_hitl_accuracy": round(post, 4),
        "hitl_accuracy_gain": round(gain, 4),
        "hitl_gain_pct":      round(gain * 100, 2),
    }
