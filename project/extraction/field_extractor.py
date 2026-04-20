import cv2

from .checkbox_extractor import CheckboxExtractor


class FieldExtractor:
    """Differential field extractor without local OCR engines."""

    def __init__(self):
        self.checkbox = CheckboxExtractor()

    def extract_fields(self, aligned_form, interaction_mask, field_definitions):
        results = []
        for fdef in field_definitions:
            fid = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])
            ih, iw = aligned_form.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, iw), min(y + h, ih)
            if x2 <= x1 or y2 <= y1:
                results.append(self._empty(fid, ftype, x, y, w, h))
                continue

            roi_form = aligned_form[y1:y2, x1:x2]
            roi_mask = interaction_mask[y1:y2, x1:x2]

            if ftype == "checkbox":
                result = self.checkbox.extract(roi_form, roi_mask)
            else:
                ink = cv2.bitwise_and(roi_form, roi_form, mask=roi_mask)
                density = float((roi_mask > 0).sum() / max(roi_mask.size, 1))
                result = {"value": "", "confidence": max(0.0, min(1.0, density)), "ink_preview": ink}

            results.append({"field_id": fid, "field_type": ftype, "x": x, "y": y, "w": w, "h": h, **result})
        return results

    @staticmethod
    def _empty(fid, ftype, x, y, w, h):
        return {
            "field_id": fid,
            "field_type": ftype,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "value": "" if ftype != "checkbox" else False,
            "confidence": 0.0,
        }
