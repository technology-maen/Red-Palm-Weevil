import os
import librosa
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier


TRAIN_FOLDER = r"sounds/"
MODEL_PATH = "sound_detector.pkl"



def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None



def train_model():
    X = []
    y = []

    print("Reading training sounds...\n")

    files = [f for f in os.listdir(TRAIN_FOLDER) if f.lower().endswith((".wav", ".mp3"))]

    if len(files) == 0:
        print(" No training audio files found in:", TRAIN_FOLDER)
        return

    for file in files:
        path = os.path.join(TRAIN_FOLDER, file)
        features = extract_features(path)
        if features is not None:
            X.append(features)
            y.append(1)
            print(f"[OK] Loaded: {file}")

    if len(X) == 0:
        print(" No valid features extracted!")
        return

    print("\nTraining model...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("\nModel trained and saved →", MODEL_PATH)


# -----------------------
# DETECT SOUND
# -----------------------
def detect_sound(test_file):
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train first.")
        return False

    model = joblib.load(MODEL_PATH)
    features = extract_features(test_file)
    if features is None:
        return False
    features = features.reshape(1, -1)
    result = model.predict(features)[0]
    return True if result == 1 else False



if __name__ == "__main__":
    print("=== TRAINING START ===")
    train_model()

    print("\nEnter path to WAV or MP3 file to test:")
    test_path = input("File path: ")

    if not os.path.exists(test_path):
        print(" File not found!")
    else:
        result = detect_sound(test_path)
        print("\nResult:", "TRUE ✓ (similar to training sounds)" if result else "FALSE ✗ (different sound)")