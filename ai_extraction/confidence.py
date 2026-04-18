import math


def levenshtein_distance(a: str, b: str) -> int:
    a = (a or "").lower()
    b = (b or "").lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _dict_confidence(extracted: str, match: str | None) -> tuple[float, int]:
    if not extracted or not match:
        return 0.0, max(len(extracted or ""), len(match or ""))
    dist = levenshtein_distance(extracted, match)
    denom = max(len(extracted), len(match), 1)
    return max(0.0, 1.0 - (dist / denom)), dist


def compute_confidence(
    extracted_value: str,
    c_lp: float,
    dictionary_match: str | None,
    critical: bool,
    w_lp: float,
    w_dict: float,
) -> tuple[float, float, int]:
    c_lp = float(max(0.0, min(1.0, c_lp)))
    if not critical or not dictionary_match:
        return c_lp, 0.0, 0

    c_dict, dist = _dict_confidence(extracted_value or "", dictionary_match)
    c_final = (w_lp * c_lp) + (w_dict * c_dict)
    return float(max(0.0, min(1.0, c_final))), float(max(0.0, min(1.0, c_dict))), int(dist)


def logprob_to_confidence(avg_logprob: float | None) -> float:
    if avg_logprob is None:
        return 0.5
    try:
        return float(max(0.0, min(1.0, math.exp(float(avg_logprob)))))
    except Exception:
        return 0.5
