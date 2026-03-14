from .semantic_validator import SemanticValidator

_ACCEPTED="accepted"; _LOW="low_confidence"; _SEM="semantic_failure"; _REJ="rejected"

class ConfidenceValidator:
    def __init__(self, confidence_threshold=0.60):
        self.threshold = confidence_threshold
        self._semantic = SemanticValidator()

    def validate(self, extracted_fields, field_definitions):
        def_map={f["id"]:f for f in field_definitions}
        results=[]
        for field in extracted_fields:
            fid=field["field_id"]; conf=float(field.get("confidence",0.0))
            value=field.get("value",""); fdef=def_map.get(fid,{})
            if field.get("field_type")!="checkbox":
                sem=self._semantic.validate(str(value),fdef)
            else:
                sem={"valid":True,"reason":"checkbox_field"}
            above=conf>=self.threshold; ok=sem["valid"]
            if above and ok:       status=_ACCEPTED;  nr=False
            elif above and not ok: status=_SEM;       nr=True
            elif not above and ok: status=_LOW;       nr=True
            else:                  status=_REJ;       nr=True
            results.append({**field,"validation_status":status,
                            "validation_reason":sem["reason"],
                            "needs_review":nr,"final_value":value,"corrected":False})
        return results
