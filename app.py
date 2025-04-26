from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

def generate_srt_for_video(video_filename, meta_path="data/meta.csv"):
    meta = pd.read_csv(meta_path)
    video_segments = meta[meta["video"] == video_filename]
    srt_lines = []
    for idx, row in video_segments.iterrows():
        start = row["start_time"]
        end = row["end_time"]
        label = row["label"]
        def srt_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        srt_lines.append(f"{len(srt_lines)+1}\n{srt_time(start)} --> {srt_time(end)}\n{label}\n")
    return "\n".join(srt_lines)

@app.route('/', methods=['GET', 'POST'])
def index():
    transcription = None
    subtitles = None
    srt_download = None
    video_url = None
    if request.method == 'POST':
        video = request.files['video']
        filename = secure_filename(video.filename)
        save_path = os.path.join('static', filename)
        video.save(save_path)
        video_url = save_path
        srt_content = generate_srt_for_video(filename)
        srt_path = os.path.join('static', filename + '.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        subtitles = srt_content
        srt_download = srt_path
        from transcribe_video import segment_and_transcribe_video, load_normalization_params, MODEL_PATH, LABELS_PATH, NORMALIZATION_PATH
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        x_min, x_max, y_min, y_max = load_normalization_params(NORMALIZATION_PATH)
        transcription = segment_and_transcribe_video(save_path, model, labels)
    return render_template('index.html', transcription=transcription, subtitles=subtitles, srt_download=srt_download, video_url=video_url)

if __name__ == '__main__':
    app.run(debug=True)