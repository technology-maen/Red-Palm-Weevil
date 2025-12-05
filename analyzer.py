"""
Analysis helper functions extracted from the Streamlit app for testing and reuse.
"""
from typing import Dict


def compute_final_score(sound_bool: bool, hole_result: Dict[str, object],
                        age_years: float, humidity_pct: float,
                        trunk_strength_input: float, exposed_crown_input: bool, wet_input: bool) -> Dict[str, object]:
    """
    Combine sound boolean and hole analysis with biases into a final score and explanation.
    Returns dict with score (0..1), category, and breakdown.
    """
    # base scores
    sound_score = 1.0 if sound_bool else 0.0
    hole_score = float(hole_result.get("score", 0.0))

    # base average (equal weights)
    base = 0.5 * sound_score + 0.5 * hole_score

    # compute bias multiplier components
    # age: older palms are more at risk -> scaled so 50 years gives +0.5 multiplier
    age_factor = min(1.0, max(0.0, age_years / 50.0))  # 0..1

    # humidity: higher humidity slightly increases risk
    humidity_factor = min(1.0, max(0.0, humidity_pct / 100.0))  # 0..1

    # trunk_strength_input: user-provided strength 0..10 => reduce risk when strong
    ts_input = max(0.0, min(10.0, trunk_strength_input))
    trunk_weakness = 1.0 - (ts_input / 10.0)  # 0..1

    exposed_factor = 1.0 if exposed_crown_input else 0.0
    wet_factor = 1.0 if wet_input else 0.0

    # combine biases into a multiplier around 1.0
    # weights tuned to produce reasonable results; clamped afterwards
    multiplier = 1.0 + (0.35 * age_factor) + (0.25 * humidity_factor) + (0.3 * trunk_weakness) + (0.25 * exposed_factor) + (0.2 * wet_factor)

    raw = base * multiplier
    score = max(0.0, min(1.0, raw))

    # categorize
    if score >= 0.75:
        category = "High risk"
    elif score >= 0.4:
        category = "Moderate risk"
    else:
        category = "Low risk"

    breakdown = {
        "sound_score": sound_score,
        "hole_score": hole_score,
        "base_average": round(base, 3),
        "age_factor": round(age_factor, 3),
        "humidity_factor": round(humidity_factor, 3),
        "trunk_weakness": round(trunk_weakness, 3),
        "exposed_factor": exposed_factor,
        "wet_factor": wet_factor,
        "multiplier": round(multiplier, 3),
        "raw": round(raw, 3),
    }

    return {"score": round(score, 3), "category": category, "breakdown": breakdown}
