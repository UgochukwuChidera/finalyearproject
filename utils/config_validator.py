from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ValidationError, field_validator

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore

if HAS_PYDANTIC:

    class BoundingBox(BaseModel):
        x: float
        y: float
        w: float
        h: float

        @field_validator("w", "h")
        @classmethod
        def positive_size(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("w/h must be > 0")
            return v

        @field_validator("x", "y")
        @classmethod
        def non_negative(cls, v: float) -> float:
            if v < 0:
                raise ValueError("x/y must be >= 0")
            return v

    class FieldDef(BaseModel):
        name: str
        type: str = "text"
        critical: bool = False
        expected_type: str = "string"
        label: str | None = None
        label_hint: str | None = None
        bounding_box: BoundingBox
        dictionary: str | None = None
        validation: dict[str, Any] | None = None
        required: bool | None = None
        regex_validation: str | None = None

        @field_validator("name")
        @classmethod
        def non_empty_name(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("field name must be non-empty")
            return v

        @field_validator("type")
        @classmethod
        def valid_type(cls, v: str) -> str:
            if v not in {"text", "checkbox", "section"}:
                raise ValueError("type must be text/checkbox/section")
            return v

    class EditorCanvas(BaseModel):
        width: float
        height: float

        @field_validator("width", "height")
        @classmethod
        def positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError("editor_canvas dimensions must be > 0")
            return v

    class ConfigSchema(BaseModel):
        template_path: str
        fields: list[FieldDef]
        editor_canvas: EditorCanvas | None = None
        form_type: str | None = None
        confidence_weights: dict[str, Any] | None = None
        thresholds: dict[str, Any] | None = None
        preprocessing: dict[str, Any] | None = None
        template_extraction: str | None = None
        bbox_normalized: bool | None = None
        coordinate_space: str | None = None

        model_config = {"extra": "allow"}

        @field_validator("template_path")
        @classmethod
        def non_empty_template(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("template_path is required")
            return v

        @field_validator("fields")
        @classmethod
        def unique_names(cls, v: list[FieldDef]) -> list[FieldDef]:
            names = [f.name for f in v]
            if len(names) != len(set(names)):
                raise ValueError("field names must be unique")
            return v

    def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = ConfigSchema.model_validate(payload)
        except ValidationError as e:
            raise ValueError(str(e)) from e
        return parsed.model_dump(mode="python")

else:

    def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("config must be a JSON object")
        if (
            not payload.get("template_path")
            or not str(payload["template_path"]).strip()
        ):
            raise ValueError("template_path is required")
        fields = payload.get("fields")
        if not isinstance(fields, list):
            raise ValueError("fields must be a list")
        seen = set()
        for idx, f in enumerate(fields):
            if not isinstance(f, dict):
                raise ValueError(f"fields[{idx}] must be an object")
            name = f.get("name")
            if not name or not str(name).strip():
                raise ValueError(f"fields[{idx}].name is required")
            if name in seen:
                raise ValueError(f"duplicate field name: {name}")
            seen.add(name)
            if f.get("type") not in (None, "text", "checkbox", "section"):
                raise ValueError(f"fields[{idx}].type must be text/checkbox/section")
            bb = f.get("bounding_box")
            if not isinstance(bb, dict):
                raise ValueError(f"fields[{idx}].bounding_box is required")
            for k in ("x", "y", "w", "h"):
                if k not in bb:
                    raise ValueError(f"fields[{idx}].bounding_box.{k} is required")
                try:
                    v = float(bb[k])
                except Exception:
                    raise ValueError(f"fields[{idx}].bounding_box.{k} must be numeric")
                if k in ("w", "h") and v <= 0:
                    raise ValueError(f"fields[{idx}].bounding_box.{k} must be > 0")
                if k in ("x", "y") and v < 0:
                    raise ValueError(f"fields[{idx}].bounding_box.{k} must be >= 0")
        ec = payload.get("editor_canvas")
        if ec is not None:
            if not isinstance(ec, dict):
                raise ValueError("editor_canvas must be an object")
            for k in ("width", "height"):
                if k in ec:
                    try:
                        v = float(ec[k])
                    except Exception:
                        raise ValueError(f"editor_canvas.{k} must be numeric")
                    if v <= 0:
                        raise ValueError(f"editor_canvas.{k} must be > 0")
        return payload
