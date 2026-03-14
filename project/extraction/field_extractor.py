import cv2
import numpy as np
from .ocr_extractor import OCRExtractor
from .checkbox_extractor import CheckboxExtractor

class FieldExtractor:
    def __init__(self, tesseract_cmd=None):
        self.ocr      = OCRExtractor(tesseract_cmd=tesseract_cmd)
        self.checkbox = CheckboxExtractor()

    def extract_fields(self, aligned_form, interaction_mask, field_definitions):
        results=[]
        for fdef in field_definitions:
            fid=fdef["id"]; ftype=fdef["type"]
            x,y,w,h=int(fdef["x"]),int(fdef["y"]),int(fdef["w"]),int(fdef["h"])
            ih,iw=aligned_form.shape[:2]
            x1,y1=max(x,0),max(y,0); x2,y2=min(x+w,iw),min(y+h,ih)
            if x2<=x1 or y2<=y1:
                results.append(self._empty(fid,ftype,x,y,w,h)); continue
            roi_form=aligned_form[y1:y2,x1:x2]
            roi_mask=interaction_mask[y1:y2,x1:x2]
            if ftype in ("printed","handwritten"):
                masked=cv2.bitwise_and(roi_form,roi_form,mask=roi_mask)
                result=self.ocr.extract(masked,ftype)
            elif ftype=="checkbox":
                result=self.checkbox.extract(roi_form,roi_mask)
            else:
                result={"value":"","confidence":0.0}
            results.append({"field_id":fid,"field_type":ftype,"x":x,"y":y,"w":w,"h":h,**result})
        return results

    @staticmethod
    def _empty(fid,ftype,x,y,w,h):
        return {"field_id":fid,"field_type":ftype,"x":x,"y":y,"w":w,"h":h,
                "value":"" if ftype!="checkbox" else False,"confidence":0.0}
