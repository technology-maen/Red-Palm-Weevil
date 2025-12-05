# Red Palm Weevil — Quick Assessment GUI

This repository contains a small Streamlit app and helper modules to combine an audio-based detector and a simple hole/trunk analyzer to estimate the risk of palm tree infestation or structural weakness.

Quick start

1. Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
# If you don't have `streamlit` already:
# pip install streamlit
```

2. Train the sound model (if you have training audio):

```bash
python sound.py
# Follow prompts to train and generate sound_detector.pkl
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

What is included

- `sound.py`: sound feature extraction and a tiny KNN classifier. Expects training audio in `sounds/` and will write `sound_detector.pkl`.
- `hole.py`: a small heuristic analyzer for hole/trunk observations.
- `analyzer.py`: contains `compute_final_score(...)` (reusable logic used by the app).
- `streamlit_app.py`: the Streamlit UI. Upload a short sound file and set observations/biases, then press Assess.

Notes and next improvements

- `hole.py` is deliberately simple. Replace or extend with image-based analysis or more sophisticated heuristics.
- Calibrate weights in `analyzer.py` with ground-truth data if available.
- Consider adding unit tests for `sound` (mocking audio) and expanding integration tests.

License: MIT-style (modify as needed)
