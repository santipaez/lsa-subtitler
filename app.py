from flask import Flask, request, jsonify, render_template
from utils import extract_frames, transcribe_frame
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    video = request.files['video']
    save_path = os.path.join("static", video.filename)
    video.save(save_path)
    
    frames = extract_frames(save_path, interval=60)
    results = [transcribe_frame(f) for f in frames]
    return jsonify({"transcription": results})

if __name__ == '__main__':
    app.run(debug=True)