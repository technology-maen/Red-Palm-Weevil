"""
Streamlit GUI integrating sound detection and image-based hole detection.

Usage: streamlit run streamlit_app.py

This app:
- accepts an optional audio file and runs `sound.detect_sound` if a trained model exists
- accepts an optional image and runs `hole.detect_holes_adaptive` to find holes
- allows manual inputs (trunk strength, wet/exposed flags, age, humidity)
- computes a final risk score using `analyzer.compute_final_score`
"""

import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from sound import detect_sound
from hole import detect_holes_adaptive, analyze_hole, print_results
from analyzer import compute_final_score


def save_upload_to_temp(uploaded, suffix="") -> Optional[str]:
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def img_to_bytes_bgr(img_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode('.jpg', img_bgr)
    return buf.tobytes()


def main():
    st.set_page_config(page_title="Red Palm Weevil — Assessment", layout="wide")
    st.title("Red Palm Weevil — Quick Assessment (Audio + Image)")

    st.markdown("Upload an optional audio clip and/or an image of the trunk/crown. Provide a few observations and press Assess.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.header("Image (optional)")
        image_file = st.file_uploader("Upload trunk/crown image", type=["png", "jpg", "jpeg"], key="img")
        image_method = st.selectbox("Detection method", options=["combined", "dark", "texture"], index=0)
        show_debug = st.checkbox("Show debug mask/image", value=False)

        st.write("---")
        st.header("Manual hole/trunk observations (overrides/augments image)")
        trunk_strength = st.slider("Trunk strength (1 weak - 10 strong)", 0.0, 10.0, 5.0, 0.5)
        hole_depth_manual = st.number_input("Hole depth (cm) — manual (0 = unknown)", min_value=0.0, value=0.0, step=1.0)
        hole_wet = st.checkbox("Hole area wet / moist", value=False)
        exposed_crown = st.checkbox("Exposed crown present", value=False)

    with col_right:
        st.header("Audio (optional)")
        audio_file = st.file_uploader("Upload short audio (wav/mp3)", type=["wav", "mp3"], key="audio")
        st.write("---")
        st.header("Biases")
        age = st.number_input("Age of palm (years)", min_value=0.0, value=10.0, step=1.0)
        humidity = st.slider("Local humidity (%)", 0, 100, 50)

        st.write("---")
        assess = st.button("Assess")

    # default outputs
    annotated_img_bytes = None
    detected_holes = []
    combined_mask = None
    debug_img = None
    sound_detected = None
    sound_msg = "No audio provided or model missing"

    if assess:
        # handle image
        img_path = save_upload_to_temp(image_file, suffix=".jpg") if image_file is not None else None
        if img_path:
            try:
                result_img, holes, mask, dbg = detect_holes_adaptive(img_path, method=image_method)
                detected_holes = holes
                combined_mask = mask
                debug_img = dbg
                annotated_img_bytes = img_to_bytes_bgr(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            except Exception as e:
                st.error(f"Image processing error: {e}")
            finally:
                try:
                    os.remove(img_path)
                except Exception:
                    pass

        # handle audio
        audio_path = save_upload_to_temp(audio_file, suffix=".wav") if audio_file is not None else None
        if audio_path:
            try:
                detected = detect_sound(audio_path)
                sound_detected = bool(detected)
                sound_msg = "Detected: TRUE (similar to training sounds)" if sound_detected else "Detected: FALSE"
            except Exception as e:
                sound_detected = False
                sound_msg = f"Sound detector error: {e}"
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

        # derive hole_depth from image if not provided manually
        hole_depth_cm = float(hole_depth_manual)
        if hole_depth_cm <= 0 and detected_holes:
            # heuristic mapping: based on number of holes and largest area ratio
            num = len(detected_holes)
            areas = [h['area'] for h in detected_holes]
            largest = max(areas) if areas else 0
            # area ratio relative to image
            # Try to estimate approximate depth: more/larger holes => deeper
            img_area = result_img.shape[0] * result_img.shape[1]
            ratio = largest / img_area if img_area > 0 else 0
            # map ratio to cm (0..30 cm)
            estimated_cm = min(30.0, max(0.0, (ratio * 1000) + (num * 2)))
            hole_depth_cm = round(estimated_cm, 1)

        # Use manual wet/exposed unless image analysis suggests otherwise (not automated yet)

        # run hole analyzer
        hole_res = analyze_hole(trunk_strength=trunk_strength, hole_depth_cm=hole_depth_cm, wet=hole_wet, exposed_crown=exposed_crown)

        # make sure sound_detected is boolean
        if sound_detected is None:
            sound_detected = False

        final = compute_final_score(sound_bool=sound_detected, hole_result=hole_res,
                                    age_years=age, humidity_pct=float(humidity),
                                    trunk_strength_input=trunk_strength,
                                    exposed_crown_input=exposed_crown, wet_input=hole_wet)

        # Show results
        st.subheader("Assessment result")
        st.metric("Risk score", f"{final['score']*100:.1f}%")
        st.write(f"Category: **{final['category']}**")

        st.write("### Breakdown")
        st.json(final['breakdown'])

        st.write("### Sound detector")
        st.write(sound_msg)

        st.write("### Hole analysis (heuristic)")
        st.json(hole_res)

        st.write("### Image detection summary")
        st.write(f"Holes found: {len(detected_holes)}")
        if len(detected_holes) > 0:
            st.write(detected_holes)

        if annotated_img_bytes is not None:
            st.write("### Annotated image")
            st.image(annotated_img_bytes, use_column_width=True)
            if show_debug and debug_img is not None:
                st.write("### Debug image / mask")
                # show mask
                st.image(img_to_bytes_bgr(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)), caption="debug (RGB)")
                if combined_mask is not None:
                    _, mask_buf = cv2.imencode('.jpg', combined_mask)
                    st.image(mask_buf.tobytes(), caption="mask")

    else:
        st.info("Set inputs and press Assess to run the estimator.")


if __name__ == '__main__':
    main()
