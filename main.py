import subprocess
import sys

def transcribe_video(video_path):
    keypoints_path = "temp_keypoints.npy"
    print(f"Extrayendo keypoints de {video_path}...")
    result = subprocess.run(
        [sys.executable, "extract_keypoints.py", video_path, keypoints_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error extrayendo keypoints:", result.stderr)
        return
    print("Realizando predicción...")
    result = subprocess.run(
        [sys.executable, "predict.py", keypoints_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Error en la predicción:", result.stderr)
        return
    pred_line = [line for line in result.stdout.splitlines() if line.startswith("Predicción:")]
    if pred_line:
        transcription = pred_line[0].replace("Predicción: ", "")
        with open("transcripcion.txt", "w", encoding="utf-8") as f:
            f.write(transcription + "\n")
        print("Transcripción guardada en transcripcion.txt")
    else:
        print("No se pudo obtener la transcripción.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_al_video>")
        print("Ejemplo: python main.py data/clips/mi_video.mp4")
        sys.exit(1)
    video_path = sys.argv[1]
    transcribe_video(video_path)