from flask import Flask, render_template, request, redirect
import os
import cv2
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "static/audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        traffic_level = analyze_video(filepath)
        audio_file = generate_voice(traffic_level)

        return render_template("index.html",
                               result=traffic_level,
                               audio=audio_file)

    return render_template("index.html", result=None, audio=None)


def analyze_video(path):
    cap = cv2.VideoCapture(path)

    total = 0
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        classes = results[0].boxes.cls.tolist()

        count = sum(1 for c in classes if int(c) in VEHICLE_CLASSES)

        total += count
        frames += 1

    cap.release()

    avg = total / max(frames, 1)

    if avg < 3:
        return "Traffic is Low"
    elif avg < 7:
        return "Traffic is Medium"
    else:
        return "Traffic is Heavy"


def generate_voice(text):
    path = os.path.join(AUDIO_FOLDER, "voice.mp3")

    tts = gTTS(text=text, lang="en")
    tts.save(path)

    return path


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)