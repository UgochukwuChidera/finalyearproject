from __future__ import annotations


class HITLEscalation:
    @staticmethod
    def get_flagged(validated_fields: list[dict]) -> list[dict]:
        return [f for f in validated_fields if f.get("needs_review", False)]

    @staticmethod
    def apply_correction(
        field_id: str, corrected_value: object, validated_fields: list[dict]
    ) -> list[dict]:
        updated: list[dict] = []
        for field in validated_fields:
            if field["field_id"] == field_id:
                field = {**field, "final_value": corrected_value, "corrected": True,
                         "validation_status": "corrected", "needs_review": False}
            updated.append(field)
        return updated

    @classmethod
    def apply_bulk_corrections(
        cls, corrections: dict[str, object], validated_fields: list[dict]
    ) -> list[dict]:
        for fid, val in corrections.items():
            validated_fields = cls.apply_correction(fid, val, validated_fields)
        return validated_fields

    @staticmethod
    def escalation_stats(validated_fields: list[dict]) -> dict[str, float | int]:
        total = len(validated_fields)
        flagged = sum(1 for f in validated_fields if f.get("needs_review", False))
        corrected = sum(1 for f in validated_fields if f.get("corrected", False))
        return {"total_fields": total, "flagged_count": flagged, "corrected_count": corrected,
                "escalation_rate": round(flagged / total, 4) if total else 0.0}
