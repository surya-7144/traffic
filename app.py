from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "static/audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Analyze video
            traffic_level = analyze_video(filepath)

            # Generate voice
            text = f"Traffic is {traffic_level}"
            audio_path = os.path.join(AUDIO_FOLDER, "voice.mp3")

            tts = gTTS(text=text, lang="en")
            tts.save(audio_path)

            return render_template(
                "index.html",
                result=traffic_level,
                audio_file=audio_path
            )

    return render_template("index.html")


def analyze_video(path):
    cap = cv2.VideoCapture(path)

    vehicle_count = 0
    frame_limit = 5   # 🔥 VERY SMALL (important)
    processed = 0

    while processed < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔽 make it very small
        frame = cv2.resize(frame, (224, 224))

        # ⚡ faster YOLO
        results = model(frame, imgsz=224, conf=0.3)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    vehicle_count += 1

        processed += 1

    cap.release()

    if vehicle_count < 5:
        return "Low"
    elif vehicle_count < 15:
        return "Medium"
    else:
        return "High"


if __name__ == "__main__":
    app.run(debug=True)
