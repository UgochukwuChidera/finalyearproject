import re

_PATTERNS = {
    "date":         re.compile(r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"),
    "email":        re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[\w\.]{2,}$"),
    "phone":        re.compile(r"^\+?[\d\s\-\(\)]{7,15}$"),
    "numeric":      re.compile(r"^-?\d+(\.\d+)?$"),
    "integer":      re.compile(r"^-?\d+$"),
    "alpha":        re.compile(r"^[A-Za-z\s\-\'\.]+$"),
    "alphanumeric": re.compile(r"^[A-Za-z0-9\s\-_\.]+$"),
}

class SemanticValidator:
    def validate(self, value, field_def):
        req    = bool(field_def.get("required",False))
        fmt    = field_def.get("format")
        maxlen = field_def.get("max_length")
        allow  = field_def.get("allowed_values")
        s = value.strip() if isinstance(value,str) else ""
        if not s:
            if req: return {"valid":False,"reason":"required_field_empty"}
            return {"valid":True,"reason":"optional_empty"}
        if fmt and fmt in _PATTERNS:
            if not _PATTERNS[fmt].match(s):
                return {"valid":False,"reason":f"format_mismatch:{fmt}"}
        if maxlen and len(s)>maxlen:
            return {"valid":False,"reason":f"exceeds_max_length:{maxlen}"}
        if allow:
            if s.lower() not in [v.lower() for v in allow]:
                return {"valid":False,"reason":"value_not_in_allowed_set"}
        return {"valid":True,"reason":"passed"}
