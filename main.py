import glob
import os
import numpy as np
from tqdm import tqdm

def segment_and_transcribe_video(video_path, model, labels, x_min, x_max, y_min, y_max, window_size=60, step_size=30):
    import cv2
    from extract_keypoints import extract_keypoints_from_video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    predictions = []
    for start in range(0, total_frames, step_size):
        keypoints = extract_keypoints_from_video(video_path, max_frames=window_size)
        keypoints[..., 0] = (keypoints[..., 0] - x_min) / (x_max - x_min + 1e-8)
        keypoints[..., 1] = (keypoints[..., 1] - y_min) / (y_max - y_min + 1e-8)
        keypoints = np.nan_to_num(keypoints, nan=0.0)
        if keypoints.shape[0] < window_size:
            pad = np.zeros((window_size, 33, 4))
            pad[:keypoints.shape[0]] = keypoints
            keypoints = pad
        keypoints = np.expand_dims(keypoints, axis=0)
        pred = model.predict(keypoints)
        idx = np.argmax(pred)
        predictions.append(labels[idx])
    transcripcion = []
    for label in predictions:
        if not transcripcion or transcripcion[-1] != label:
            transcripcion.append(label)
    return " ".join(transcripcion)

def transcribe_videos_from_folder_segmented(video_folder, model_path, labels_path, output_path, window_size=60, step_size=30):
    from tensorflow.keras.models import load_model
    import cv2
    model = load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    from train import x_min, x_max, y_min, y_max
    transcripciones = {}
    for video in tqdm(video_files, desc="Transcribiendo videos"):
        transcripcion = segment_and_transcribe_video(
            video, model, labels, x_min, x_max, y_min, y_max, window_size, step_size
        )
        transcripciones[os.path.basename(video)] = transcripcion
    with open(output_path, "w", encoding="utf-8") as f:
        for video, trans in transcripciones.items():
            f.write(f"{video}: {trans}\n")
    print(f"Transcripción segmentada guardada en {output_path}")

if __name__ == "__main__":
    transcribe_videos_from_folder_segmented(
        video_folder="data/clips",
        model_path="lsa_sign_model.keras",
        labels_path="labels.txt",
        output_path="transcription.txt",
        window_size=60,
        step_size=30
    )