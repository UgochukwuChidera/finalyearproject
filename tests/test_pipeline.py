"""Integration-style tests for the pipeline module.

Tests cover: config loading, template extraction, deskewing, cropping,
validation helpers, checkbox detection, and the top-level process_form
orchestrator with mocked AI extraction calls.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from pipeline import (
    _load_config,
    _load_template_extraction,
    _to_png_bytes,
    _safe_crop,
    _deskew_image,
    _validation_ok,
    _is_checkbox,
    _differs_from_template,
    process_form,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_by_name(self, mock_config_path: str):
        """Loading a valid config by path should return (dict, path_str)."""
        config, resolved = _load_config(config_path=mock_config_path)
        assert isinstance(config, dict)
        assert "form_type" in config
        assert "fields" in config
        assert resolved == mock_config_path

    def test_process_form_invalid_config(self, tmp_path: Path):
        """Non-existent config → graceful error (FileNotFoundError)."""
        bad_path = str(tmp_path / "does_not_exist.json")
        with pytest.raises(FileNotFoundError):
            _load_config(config_path=bad_path)

    def test_load_config_no_name_no_path(self):
        """Calling with neither config_name nor config_path → ValueError."""
        with pytest.raises(ValueError):
            _load_config()


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

class TestLoadTemplateExtraction:
    def test_load_template_extraction_missing_path(self):
        """If config has no template_extraction path, return empty dict."""
        result = _load_template_extraction({})
        assert result == {}

    def test_load_template_extraction_nonexistent(self, tmp_path: Path):
        """If the template_extraction path doesn't exist, return empty dict."""
        config = {"template_extraction": str(tmp_path / "nonexistent.json")}
        result = _load_template_extraction(config)
        assert result == {}
    
    def test_load_template_extraction_valid(self, tmp_path: Path):
        """Valid template_extraction JSON should return the fields dict."""
        p = tmp_path / "template.json"
        with open(str(p), "w") as f:
            json.dump({"fields": {"name": {"type": "string"}}}, f)
        config = {"template_extraction": str(p)}
        result = _load_template_extraction(config)
        assert result == {"name": {"type": "string"}}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestToPngBytes:
    def test_to_png_bytes_valid(self):
        """A valid numpy array → non-empty bytes."""
        img = np.ones((10, 10), dtype=np.uint8) * 128
        data = _to_png_bytes(img)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_to_png_bytes_single_pixel(self):
        """A 1x1 array produces non-empty bytes."""
        img = np.zeros((1, 1), dtype=np.uint8)
        data = _to_png_bytes(img)
        assert isinstance(data, bytes)
        assert len(data) > 0


class TestSafeCrop:
    def test_safe_crop_normal(self):
        """Normal crop within bounds → returns expected region."""
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        crop = _safe_crop(img, 2, 2, 4, 4)
        assert crop.shape == (4, 4)

    def test_safe_crop_out_of_bounds(self):
        """Crop extending beyond image boundaries → clamped gracefully."""
        img = np.ones((10, 10), dtype=np.uint8)
        crop = _safe_crop(img, 8, 8, 10, 10)
        assert crop.size > 0

    def test_safe_crop_empty_input(self):
        """None input → placeholder 1x1 array."""
        crop = _safe_crop(None, 0, 0, 10, 10)
        assert crop.shape == (1, 1)


class TestDeskewImage:
    def test_deskew_image(self):
        """Deskewing a synthetic image should return same shape."""
        img = np.ones((100, 100), dtype=np.uint8) * 200
        img[20:80, 20:80] = 50
        deskewed = _deskew_image(img, 3.0)
        assert deskewed.shape == img.shape
        assert deskewed.dtype == img.dtype

    def test_deskew_image_none(self):
        """None input → returns None (no crash)."""
        assert _deskew_image(None, 3.0) is None


class TestValidationOk:
    def test_validation_min_length_ok(self):
        """String meeting min_length → (True, 'passed')."""
        ok, reason = _validation_ok("Hello", {"validation": {"min_length": 3}})
        assert ok is True
        assert reason == "passed"

    def test_validation_min_length_fail(self):
        """String too short → (False, reason)."""
        ok, reason = _validation_ok("Hi", {"validation": {"min_length": 5}})
        assert ok is False
        assert "min_length" in reason

    def test_validation_regex_ok(self):
        """Value matching regex → (True, 'passed')."""
        ok, reason = _validation_ok("abc@example.com", {"validation": {"regex": r".+@.+\..+"}})
        assert ok is True

    def test_validation_regex_fail(self):
        """Value not matching regex → (False, reason)."""
        ok, reason = _validation_ok("notanemail", {"validation": {"regex": r".+@.+\..+"}})
        assert ok is False

    def test_validation_numeric_range_ok(self):
        """Numeric value within range → (True, 'passed')."""
        ok, reason = _validation_ok("25", {"validation": {"min": 18, "max": 99}})
        assert ok is True

    def test_validation_no_rules(self):
        """No validation rules → always passes."""
        ok, reason = _validation_ok("anything", {"validation": {}})
        assert ok is True

    def test_validation_none_value(self):
        """None value with min_length → fails."""
        ok, reason = _validation_ok(None, {"validation": {"min_length": 1}})
        assert ok is False


class TestIsCheckbox:
    def test_is_checkbox_true(self):
        """'checkbox_group', 'checkbox', 'boolean' → True."""
        assert _is_checkbox({"expected_type": "checkbox_group"}) is True
        assert _is_checkbox({"expected_type": "checkbox"}) is True
        assert _is_checkbox({"expected_type": "boolean"}) is True

    def test_is_checkbox_false(self):
        """Other types → False."""
        assert _is_checkbox({"expected_type": "string"}) is False
        assert _is_checkbox({"expected_type": "date"}) is False
        assert _is_checkbox({}) is False


class TestDiffersFromTemplate:
    def test_differs_true(self):
        """Value different from baseline → True."""
        assert _differs_from_template("Smith", "") is True

    def test_differs_false(self):
        """Value matches baseline → False."""
        assert _differs_from_template("Smith", "Smith") is False

    def test_differs_none_value(self):
        """None value → False (no diff)."""
        assert _differs_from_template(None, "baseline") is False


# ---------------------------------------------------------------------------
# process_form (mocked AI extraction)
# ---------------------------------------------------------------------------

class TestProcessForm:
    @patch("pipeline.GeminiClient")
    @patch("pipeline.DictionaryStore")
    @patch("pipeline.TemplateAligner")
    @patch("pipeline.DifferentialAnalyzer")
    @patch("pipeline.DataExporter")
    @patch("pipeline.RelationalXLSXExporter")
    @patch("pipeline.AuditLogger")
    @patch("pipeline.OutputStructurer")
    def test_process_form_minimal(
        self,
        mock_structurer_cls: MagicMock,
        mock_audit_cls: MagicMock,
        mock_xlsx_cls: MagicMock,
        mock_exporter_cls: MagicMock,
        mock_differ_cls: MagicMock,
        mock_aligner_cls: MagicMock,
        mock_dict_store_cls: MagicMock,
        mock_gemini_cls: MagicMock,
        tmp_path: Path,
        mock_config_path: str,
    ):
        """Run process_form with a synthetic image and mocked AI → returns expected structure."""
        # --- Create a synthetic image ---
        img_path = str(tmp_path / "test_form.png")
        img = np.ones((200, 200, 3), dtype=np.uint8) * 240
        img[50:150, 50:150] = (60, 60, 60)
        cv2.imwrite(img_path, img)

        # --- Mock template image ---
        template_path = str(tmp_path / "template.png")
        cv2.imwrite(template_path, np.ones((200, 200), dtype=np.uint8) * 255)
        config_path_for_test = str(tmp_path / "test_config.json")
        config_data = {
            "form_type": "test_form",
            "fields": [
                {
                    "name": "full_name",
                    "expected_type": "string",
                    "critical": True,
                    "bounding_box": {"x": 10, "y": 10, "w": 100, "h": 20},
                    "dictionary": None,
                    "validation": {},
                },
            ],
            "template_path": template_path,
            "confidence_weights": {"w_lp": 0.6, "w_dict": 0.4},
            "thresholds": {"auto_accept": 0.85, "review": 0.70},
            "preprocessing": {"deskew": False},
            "editor_canvas": {"width": 800, "height": 1100},
        }
        with open(config_path_for_test, "w") as f:
            json.dump(config_data, f)

        # --- Mock AI extraction ---
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.extract_from_images.return_value = {
            "fields": {"full_name": "John Doe"},
            "meta": {
                "has_logprobs": False,
                "C_lp": {"full_name": 0.9},
                "overall_confidence": 0.85,
                "raw_response_preview": "...",
            },
        }
        mock_gemini_cls.return_value = mock_client

        # --- Mock template aligner ---
        mock_aligner = MagicMock()
        mock_aligner.align.return_value = (
            np.ones((200, 200), dtype=np.uint8) * 200,
            np.eye(3),
            {"confidence": 0.9, "matches": 10, "inliers": 9, "error": 0.5},
        )
        mock_aligner_cls.return_value = mock_aligner

        # --- Mock differential analyzer ---
        mock_differ = MagicMock()
        mock_differ.analyze.return_value = (
            np.zeros((200, 200), dtype=np.uint8),
            {"diff_area": 0, "diff_count": 0, "max_diff_size": 0},
        )
        mock_differ_cls.return_value = mock_differ

        # --- Mock dictionary store ---
        mock_store = MagicMock()
        mock_store.load.return_value = {}
        mock_dict_store_cls.return_value = mock_store

        # --- Mock exporter ---
        mock_exporter = MagicMock()
        mock_exporter.export_all.return_value = {
            "json": str(tmp_path / "out.json"),
            "csv": str(tmp_path / "out.csv"),
        }
        mock_exporter_cls.return_value = mock_exporter

        mock_xlsx = MagicMock()
        mock_xlsx.export.return_value = str(tmp_path / "out.xlsx")
        mock_xlsx_cls.return_value = mock_xlsx

        # --- Mock audit logger ---
        mock_audit = MagicMock()
        mock_audit.log.return_value = str(tmp_path / "audit.jsonl")
        mock_audit_cls.return_value = mock_audit

        # --- Mock structurer ---
        mock_structurer = MagicMock()
        mock_structurer.structure.return_value = {
            "fields": [],
            "metadata": {"job_id": "test", "form_type": "test"},
        }
        mock_structurer_cls.return_value = mock_structurer

        # --- Execute ---
        result = process_form(
            image_path=img_path,
            config_path=config_path_for_test,
            output_dir=str(tmp_path / "outputs"),
            log_dir=str(tmp_path / "logs"),
            dictionaries_dir=str(tmp_path / "dictionaries"),
            dpi=300,
            job_id="test_job_001",
        )

        # --- Assert ---
        assert isinstance(result, dict)
        assert result["job_id"] == "test_job_001"
        assert result["status"] in ("completed", "pending_review")
        assert "fields" in result
        assert "structured_output" in result
        assert "export_paths" in result

    def test_process_form_invalid_image(self, mock_config_path: str):
        """Non-existent image → graceful error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            process_form(
                image_path="/nonexistent/image.png",
                config_path=mock_config_path,
            )
