"""
Base Pipeline
=============
Abstract base class all three evaluation pipelines must implement.

Every pipeline returns fields in a unified format so that the same
ConfidenceValidator, HITLInterface, and Evaluator can operate on all
three without modification.

Standard field dict
-------------------
{
    "field_id"   : str,
    "field_type" : "printed" | "handwritten" | "checkbox",
    "value"      : str | bool,
    "confidence" : float,         # [0, 1]
    "x", "y", "w", "h" : int,    # bounding box from template
}
"""

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """
    All pipelines must implement extract() and expose a name property.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pipeline name used in result tables."""

    @abstractmethod
    def extract(
        self,
        image_path: str,
        template_id: str,
    ) -> tuple[list[dict], dict]:
        """
        Extract all fields from a scanned completed form.

        Parameters
        ----------
        image_path  : path to the scanned form image
        template_id : key used to look up field definitions in registry

        Returns
        -------
        fields : list[dict] — one entry per field in unified format
        stats  : dict       — pipeline-specific metadata / timings
        """
