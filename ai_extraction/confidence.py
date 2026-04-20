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


def compute_C_lp(logprobs_data) -> float:
    if logprobs_data is None:
        return 0.5
    if isinstance(logprobs_data, (int, float)):
        try:
            v = float(logprobs_data)
            if v <= 0:
                v = math.exp(v)
            return float(max(0.0, min(1.0, v)))
        except Exception:
            return 0.5

    try:
        values = [float(v) for v in logprobs_data if v is not None]
        if not values:
            return 0.5
        avg = sum(values) / len(values)
        return float(max(0.0, min(1.0, math.exp(avg))))
    except Exception:
        return 0.5


def compute_C_final(C_lp: float, C_dict: float, w_lp: float, w_dict: float) -> float:
    c_final = (float(w_lp) * float(C_lp)) + (float(w_dict) * float(C_dict))
    return float(max(0.0, min(1.0, c_final)))


def logprob_to_confidence(avg_logprob: float | None) -> float:
    return compute_C_lp(avg_logprob)
