"""
Streamlit GUI for combining sound detection and hole analysis to estimate risk.

Usage: streamlit run streamlit_app.py

This app uses `sound.detect_sound(test_file)` (expects a trained model `sound_detector.pkl` in repo root)
and `hole.analyze_hole(...)` (a simple placeholder implemented in `hole.py`).

The final risk score is a biased combination of sound result and hole analysis plus user-provided
bias factors (age, humidity, trunk strength, exposed crown, wetness).
"""

import tempfile
import os
from typing import Dict

import streamlit as st

from sound import detect_sound
from hole import analyze_hole
from analyzer import compute_final_score


def save_uploaded_file(uploaded, suffix="") -> str:
    if uploaded is None:
        return ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def main():
    st.title("Red Palm Weevil — Quick Assessment")

    st.markdown("Upload a short audio clip (wav/mp3) that captures trunk/wood sounds, and provide hole/trunk observations.")

    col1, col2 = st.columns(2)

    with col1:
        audio_file = st.file_uploader("Sound file (WAV/MP3)", type=["wav", "mp3"], key="audio")
        st.caption("Optional: used to detect insect-like sounds with the trained model")

        st.write("---")
        st.header("Hole / Trunk observations")
        trunk_strength = st.slider("Trunk strength (1 = weak, 10 = very strong)", 0.0, 10.0, 5.0, 0.5)
        hole_depth = st.number_input("Hole depth (cm)", min_value=0.0, value=0.0, step=1.0)
        hole_wet = st.checkbox("Hole area wet / moist", value=False)
        exposed_crown = st.checkbox("Exposed crown present", value=False)

    with col2:
        st.header("Bias factors")
        age = st.number_input("Age of palm (years)", min_value=0.0, value=10.0, step=1.0)
        humidity = st.slider("Ambient / local humidity (%)", 0, 100, 50)

        st.write("---")
        st.header("Run")
        run_btn = st.button("Assess")

    sound_result = None
    sound_result_text = "No audio provided or model missing"

    if run_btn:
        # save audio and run detection if provided
        audio_path = save_uploaded_file(audio_file, suffix=".wav") if audio_file is not None else ""
        if audio_path:
            try:
                detected = detect_sound(audio_path)
                sound_result = bool(detected)
                sound_result_text = "Detected: TRUE (similar to training sounds)" if sound_result else "Detected: FALSE"
            except Exception as e:
                sound_result = False
                sound_result_text = f"Error running sound detector: {e}"
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

        # get hole analysis
        hole_res = analyze_hole(trunk_strength=trunk_strength, hole_depth_cm=hole_depth, wet=hole_wet, exposed_crown=exposed_crown)

        final = compute_final_score(sound_bool=bool(sound_result), hole_result=hole_res,
                                    age_years=age, humidity_pct=float(humidity),
                                    trunk_strength_input=trunk_strength,
                                    exposed_crown_input=exposed_crown, wet_input=hole_wet)

        st.subheader("Result")
        st.metric("Risk score", f"{final['score'] * 100:.1f}%", delta=None)
        st.write(f"Category: **{final['category']}**")

        st.write("### Breakdown")
        st.write(final['breakdown'])

        st.write("### Sound detector")
        st.write(sound_result_text)

        st.write("### Hole analysis details")
        st.write(hole_res)

        st.write("---")
        st.caption("Notes: This is a heuristic estimator combining a sound model and simple trunk/hole heuristics. For production use, improve the hole analysis, train/validate the sound model, and calibrate bias weights.")

    else:
        st.info("Provide inputs and press 'Assess' to run the estimator.")


if __name__ == '__main__':
    main()
