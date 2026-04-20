"""
DAPE Pipeline Orchestrator
===========================
Executes the full pipeline:
  1. Preprocessing       (DPI-aware kernel scaling)
  2. Template Alignment
  3. Differential Analysis
  4. Field Extraction
  5. Confidence Validation
  6. HITL Escalation
  7. Output Structuring
  8. Export and Audit Log
"""

from pathlib import Path

from .preprocessing.io               import load_image
from .preprocessing.grayscale        import to_grayscale
from .preprocessing.baseline_metrics import baseline_metrics
from .preprocessing.skew_analysis    import skew_analysis
from .preprocessing.illumination     import illumination_normalization
from .preprocessing.binarization     import binarization
from .preprocessing.border_removal   import border_removal
from .preprocessing.structure_prep   import structure_prep
from .preprocessing.fusion           import fuse
from .preprocessing.dpi              import kernel_sizes

from .template_registry               import TemplateRegistry
from .alignment.aligner               import TemplateAligner
from .differential.analyzer           import DifferentialAnalyzer
from .extraction.field_extractor      import FieldExtractor
from .validation.confidence_validator import ConfidenceValidator
from .hitl.escalation                 import HITLEscalation
from .hitl.interface                  import HITLInterface
from .output.structurer               import OutputStructurer
from .output.relational_exporter      import RelationalXLSXExporter
from .output.exporter                 import DataExporter
from .output.audit_logger             import AuditLogger


class DAPEOrchestrator:
    def __init__(
        self,
        registry_path        = "templates/registry.json",
        output_dir           = "outputs",
        log_dir              = "logs",
        confidence_threshold = 0.60,
        enable_hitl          = True,
        hitl_host            = "127.0.0.1",
        hitl_port            = 5050,
        dpi                  = 300,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._dpi      = dpi
        self._ks       = kernel_sizes(dpi)   # all kernels scaled to actual DPI

        self._registry   = TemplateRegistry(registry_path)
        self._aligner    = TemplateAligner(
            max_features = self._ks["max_features"]
        )
        self._differ     = DifferentialAnalyzer(
            diff_threshold  = self._ks["diff_threshold"],
            min_region_area = self._ks["min_region_area"],
        )
        self._extractor  = FieldExtractor()
        self._validator  = ConfidenceValidator(confidence_threshold)
        self._escalation = HITLEscalation()
        self._exporter   = DataExporter()
        self._xlsx       = RelationalXLSXExporter()
        self._logger     = AuditLogger(log_dir)

        self._enable_hitl = enable_hitl
        self._hitl_ui = HITLInterface(hitl_host, hitl_port) if enable_hitl else None

    def process(self, image_path, template_id, form_id=None):
        form_id = form_id or Path(image_path).stem
        stats: dict = {}
        images: dict = {}

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        image, h, w, aspect = load_image(image_path)
        gray = to_grayscale(image)
        stats.update({"original_width": w, "original_height": h,
                       "aspect_ratio": aspect, "dpi": self._dpi})
        stats.update(baseline_metrics(gray))
        stats.update(skew_analysis(
            gray, hough_threshold=self._ks["skew_hough_threshold"]
        ))
        normalized, illum = illumination_normalization(
            gray, stats["grayscale_std"],
            blur_size  = self._ks["illumination_blur"],
            clahe_tile = self._ks["clahe_tile"],
        )
        stats.update(illum)
        binary, bin_stats = binarization(
            normalized, block_size=self._ks["binarization_block"]
        )
        stats.update(bin_stats)
        cropped, crop_stats = border_removal(binary, stats["threshold_stability"])
        stats.update(crop_stats)
        template_size = self._get_template_size(template_id)
        stats.update(structure_prep(cropped, template_size))
        stats["fusion_score"] = fuse(stats)
        images.update({"gray": gray, "normalized": normalized,
                        "binary": binary, "cropped_binary": cropped})

        # ── Stage 2: Template Alignment ───────────────────────────────────────
        template_img  = self._registry.get_template_image(template_id)
        field_defs    = self._registry.get_field_definitions(template_id)
        output_schema = self._registry.get_output_schema(template_id)

        aligned, transform, align_meta = self._aligner.align(gray, template_img)
        stats.update({f"align_{k}": v for k, v in align_meta.items()})
        images["aligned"] = aligned

        # ── Stage 3: Differential Analysis ───────────────────────────────────
        interaction_mask, diff_meta = self._differ.analyze(aligned, template_img)
        stats.update({f"diff_{k}": v for k, v in diff_meta.items()
                       if not hasattr(v, "shape")})
        images["interaction_mask"] = interaction_mask

        # ── Stage 4: Field Extraction ─────────────────────────────────────────
        extracted_fields = self._extractor.extract_fields(
            aligned, interaction_mask, field_defs
        )

        # ── Stage 5: Confidence Validation ────────────────────────────────────
        validated_fields = self._validator.validate(extracted_fields, field_defs)

        # ── Stage 6: HITL ─────────────────────────────────────────────────────
        flagged = self._escalation.get_flagged(validated_fields)
        stats["hitl_flagged_count"] = len(flagged)
        if flagged and self._enable_hitl and self._hitl_ui:
            corrections  = self._hitl_ui.run_review(flagged)
            validated_fields = self._escalation.apply_bulk_corrections(
                corrections, validated_fields
            )
        esc = self._escalation.escalation_stats(validated_fields)
        stats.update({f"esc_{k}": v for k, v in esc.items()})

        # ── Stage 7: Output Structuring ───────────────────────────────────────
        structurer        = OutputStructurer(output_schema)
        structured_output = structurer.structure(
            validated_fields, form_id, template_id, stats
        )

        # ── Stage 8: Export + Audit ───────────────────────────────────────────
        base          = str(self.output_dir / form_id)
        export_paths  = self._exporter.export_all(structured_output, base)
        export_paths["xlsx"] = self._xlsx.export(structured_output, base + ".xlsx")
        audit_log     = self._logger.log(
            form_id, template_id, stats, validated_fields, export_paths
        )

        return {"structured_output": structured_output,
                "export_paths":      export_paths,
                "audit_log_path":    audit_log,
                "stats":             stats,
                "images":            images}

    def process_batch(self, image_paths, template_id):
        results = []
        for path in image_paths:
            form_id = Path(path).stem
            try:
                result = self.process(path, template_id, form_id)
            except Exception as exc:
                result = {"form_id": form_id, "error": str(exc),
                          "structured_output": None, "export_paths": {},
                          "audit_log_path": None, "stats": {}, "images": {}}
            results.append(result)
        return results

    def _get_template_size(self, template_id):
        import cv2
        entry = self._registry.get_entry(template_id)
        img   = cv2.imread(entry["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (2480, 3508)
        h, w = img.shape[:2]
        return (w, h)
