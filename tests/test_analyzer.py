import pytest

from hole import analyze_hole
from analyzer import compute_final_score


def test_hole_strong_no_hole():
    res = analyze_hole(trunk_strength=9, hole_depth_cm=0, wet=False, exposed_crown=False)
    assert res['score'] < 0.3
    assert res['weak'] is False


def test_hole_deep_wet():
    res = analyze_hole(trunk_strength=2, hole_depth_cm=30, wet=True, exposed_crown=True)
    assert res['score'] > 0.6
    assert res['weak'] is True


def test_compute_score_sound_true_hole_high():
    hole_res = {'score': 0.9}
    final = compute_final_score(sound_bool=True, hole_result=hole_res, age_years=30, humidity_pct=80, trunk_strength_input=3, exposed_crown_input=True, wet_input=True)
    assert final['score'] >= 0.75
    assert final['category'] == 'High risk'


def test_compute_score_sound_false_hole_low():
    hole_res = {'score':0.1}
    final = compute_final_score(sound_bool=False, hole_result=hole_res, age_years=5, humidity_pct=30, trunk_strength_input=9, exposed_crown_input=False, wet_input=False)
    assert final['score'] < 0.4
    assert final['category'] == 'Low risk'
