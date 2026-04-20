"""
TemplateRegistryBuilder — auto-detects fields from a blank template image.
Uses morphological operations + ORB field geometry detection.
"""
import json
from pathlib import Path
import cv2

class TemplateRegistryBuilder:
    def __init__(self, checkbox_max_area=3000, textbox_min_area=3000,
                 line_min_width=60, padding=4):
        self.checkbox_max_area = checkbox_max_area
        self.textbox_min_area  = textbox_min_area
        self.line_min_width    = line_min_width
        self.padding           = padding

    def build(self, template_id, image_path, registry_path="templates/registry.json", save=True):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")
        # Detect horizontal lines (underline fields) and bordered boxes
        fields = self._detect_fields(gray)
        output_schema = {f["id"]: f["id"] for f in fields}
        entry = {"image_path": str(image_path), "fields": fields, "output_schema": output_schema}
        if save:
            self._write(template_id, entry, registry_path)
        print(f"[RegistryBuilder] {template_id}: {len(fields)} fields detected")
        return entry

    def _detect_fields(self, gray):
        import numpy as np
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        _, binary = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Horizontal lines
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT,(self.line_min_width,1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k)
        # Vertical lines
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k)
        combined = cv2.bitwise_or(h_lines, v_lines)
        close_k  = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        closed   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
        contours,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ih, iw = gray.shape[:2]
        fields = []; p = self.padding
        for idx, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            if area > 0.85*ih*iw or area < 100:
                continue
            ftype = "checkbox" if area <= self.checkbox_max_area else "handwritten"
            fields.append({"id": f"field_{idx+1:03d}", "type": ftype,
                            "x": max(0,x-p), "y": max(0,y-p),
                            "w": min(iw-x+p, w+2*p), "h": min(ih-y+p, h+2*p)})
        fields.sort(key=lambda r:(r["y"]//20,r["x"]))
        return fields

    @staticmethod
    def _write(template_id, entry, registry_path):
        path = Path(registry_path); path.parent.mkdir(parents=True, exist_ok=True)
        registry = {}
        if path.exists():
            try:
                with path.open() as fh: registry = json.load(fh)
            except: pass
        registry[template_id] = entry
        with path.open("w") as fh:
            json.dump(registry, fh, indent=2, ensure_ascii=False)
        print(f"[RegistryBuilder] Registry saved → {registry_path}")
