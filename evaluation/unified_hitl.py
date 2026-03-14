"""
Unified HITL Controller
========================
Applies exactly the same confidence-based validation and human-in-the-loop
escalation to all three pipeline outputs.

This ensures that the HITL comparison is fair: the same threshold,
the same review interface, and the same correction logic regardless
of which pipeline produced the extracted fields.

Stages applied here (same for all pipelines):
  1. ConfidenceValidator  — flags fields below threshold or semantic rules
  2. HITLInterface        — Flask review UI (if enabled and fields flagged)
  3. HITLEscalation       — applies human corrections to final_value

The result is a validated field list suitable for metric computation.
"""

from project.validation.confidence_validator import ConfidenceValidator
from project.hitl.escalation                 import HITLEscalation
from project.hitl.interface                  import HITLInterface


class UnifiedHITL:
    """
    Wraps ConfidenceValidator + HITLInterface + HITLEscalation into one
    call that works identically for all three pipelines.

    Parameters
    ----------
    confidence_threshold : float
        Fields with confidence below this are flagged for review.
    enable_hitl : bool
        False → validation only (no human review). Used for pre-HITL metrics.
    hitl_host, hitl_port : str, int
        Flask server address for the review UI.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        enable_hitl:          bool  = True,
        hitl_host:            str   = "127.0.0.1",
        hitl_port:            int   = 5050,
    ):
        self._validator  = ConfidenceValidator(confidence_threshold)
        self._escalation = HITLEscalation()
        self._threshold  = confidence_threshold
        self._enable_hitl = enable_hitl

        self._hitl_ui: HITLInterface | None = None
        if enable_hitl:
            self._hitl_ui = HITLInterface(hitl_host, hitl_port)

    def run(
        self,
        extracted_fields: list[dict],
        field_definitions: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Validate fields, optionally run human review, return final fields.

        Parameters
        ----------
        extracted_fields  : output of any pipeline's extract()
        field_definitions : template field defs (for semantic validation)

        Returns
        -------
        validated_fields : list[dict] — with validation_status, final_value, corrected
        hitl_stats       : dict       — flagged/corrected counts
        """
        # Stage 1: confidence + semantic validation
        validated = self._validator.validate(extracted_fields, field_definitions)

        flagged = self._escalation.get_flagged(validated)

        # Stage 2: optional human review
        if flagged and self._enable_hitl and self._hitl_ui is not None:
            corrections = self._hitl_ui.run_review(flagged)
            validated   = self._escalation.apply_bulk_corrections(
                corrections, validated
            )

        hitl_stats = self._escalation.escalation_stats(validated)
        return validated, hitl_stats

    def validate_only(
        self,
        extracted_fields:  list[dict],
        field_definitions: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Run validation without HITL (used to capture pre-HITL metrics).
        Always safe to call regardless of enable_hitl setting.
        """
        validated  = self._validator.validate(extracted_fields, field_definitions)
        hitl_stats = self._escalation.escalation_stats(validated)
        return validated, hitl_stats
