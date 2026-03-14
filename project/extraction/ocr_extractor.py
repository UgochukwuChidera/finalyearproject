import cv2
import numpy as np
import pytesseract

class OCRExtractor:
    _CFG = "--oem 3 --psm 7"

    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, field_image, field_type="printed"):
        _E = {"value":"","confidence":0.0,"raw_data":{"words":[],"confidences":[]}}
        if field_image is None or field_image.size==0: return _E
        if len(field_image.shape)==3:
            field_image = cv2.cvtColor(field_image,cv2.COLOR_BGR2GRAY)
        try:
            data = pytesseract.image_to_data(field_image,config=self._CFG,
                                             output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract not found. Install it and ensure it is on PATH.")
        words,confs=[],[]
        for text,conf in zip(data["text"],data["conf"]):
            text=str(text).strip(); conf=int(conf)
            if text and conf>=0: words.append(text); confs.append(conf)
        value=" ".join(words).strip()
        avg=float(np.mean(confs)/100.0) if confs else 0.0
        return {"value":value,"confidence":float(np.clip(avg,0,1)),
                "raw_data":{"words":words,"confidences":confs}}
