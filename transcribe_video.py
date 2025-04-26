import sys
import numpy as np
import tensorflow as tf
from extract_keypoints import extract_keypoints_from_video

# Cargar modelo y labels
MODEL_PATH = "lsa_sign_model.keras"
LABELS_PATH = "labels.txt"
TRANSCRIPTION_PATH = "transcription.txt"
NORMALIZATION_PATH = "normalization.txt"

x_min = 0.0 
y_min = 0.0
x_max = 1.0
y_max = 1.0
WINDOW_SIZE = 60
STEP_SIZE = 30

def load_normalization_params(path):
    with open(path, "r", encoding="utf-8") as f:
        vals = f.read().strip().split()
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

def normalize_keypoints(keypoints):
    keypoints = keypoints.copy()
    keypoints[..., 0] = (keypoints[..., 0] - x_min) / (x_max - x_min + 1e-8)
    keypoints[..., 1] = (keypoints[..., 1] - y_min) / (y_max - y_min + 1e-8)
    keypoints = np.nan_to_num(keypoints, nan=0.0)
    return keypoints

def segment_and_transcribe_video(video_path, model, labels):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    predictions = []
    for start in range(0, total_frames, STEP_SIZE):
        keypoints = extract_keypoints_from_video(video_path, max_frames=WINDOW_SIZE)
        keypoints = normalize_keypoints(keypoints)
        if keypoints.shape[0] < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE, 33, 4))
            pad[:keypoints.shape[0]] = keypoints
            keypoints = pad
        keypoints = np.expand_dims(keypoints, axis=0)
        pred = model.predict(keypoints, verbose=0)
        idx = np.argmax(pred)
        predictions.append(labels[idx])
    transcripcion = []
    for label in predictions:
        if not transcripcion or transcripcion[-1] != label:
            transcripcion.append(label)
    return " ".join(transcripcion)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python transcribe_video.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    try:
        x_min, x_max, y_min, y_max = load_normalization_params(NORMALIZATION_PATH)
    except Exception as e:
        print(f"No se pudo leer {NORMALIZATION_PATH}. Usando valores por defecto. Error: {e}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Procesando video: {video_path}")
    transcription = segment_and_transcribe_video(video_path, model, labels)
    with open(TRANSCRIPTION_PATH, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcripción guardada en {TRANSCRIPTION_PATH}")
