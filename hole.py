"""
Simple hole analyzer placeholder.
Provides analyze_hole(...) which estimates a hole-based risk score [0..1]
This is intentionally conservative and easy to extend later.
"""

from typing import Dict


def analyze_hole(trunk_strength: float = 5.0,
                 hole_depth_cm: float = 0.0,
                 wet: bool = False,
                 exposed_crown: bool = False) -> Dict[str, object]:
    """
    Analyze simple hole/trunk/crown observations and return a result dict.

    Inputs:
    - trunk_strength: 0..10 (10 = very strong, 0 = collapsed)
    - hole_depth_cm: depth of the detected hole in centimeters
    - wet: whether the hole area / trunk looks wet
    - exposed_crown: whether the crown is visibly exposed

    Returns a dict with:
    - "score": float between 0 and 1 (higher => higher probability of infestation/weak tree)
    - "weak": bool (True if the tree is judged weak by heuristics)
    - "details": small dict with contributing factors
    """
    # normalize trunk strength
    ts = max(0.0, min(10.0, float(trunk_strength)))
    trunk_factor = 1.0 - (ts / 10.0)  # 0 if very strong, 1 if absent

    # hole factor: deeper hole = more concerning. assume >10cm significant
    hd = max(0.0, float(hole_depth_cm))
    hole_factor = min(1.0, hd / 20.0)  # 0..1 (20cm or more => 1)

    wet_factor = 1.0 if wet else 0.0
    exposed_factor = 1.0 if exposed_crown else 0.0

    # combine simple weighted sum
    score = (0.45 * hole_factor) + (0.35 * trunk_factor) + (0.1 * wet_factor) + (0.1 * exposed_factor)
    score = max(0.0, min(1.0, score))

    # simple boolean weak judgement
    weak = (ts < 4.0) or (hole_factor > 0.5 and wet)

    details = {
        "trunk_strength": ts,
        "trunk_factor": round(trunk_factor, 3),
        "hole_depth_cm": hd,
        "hole_factor": round(hole_factor, 3),
        "wet": wet,
        "exposed_crown": exposed_crown,
    }

    return {"score": round(score, 3), "weak": weak, "details": details}
