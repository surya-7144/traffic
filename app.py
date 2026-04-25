from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

            return render_template(
                "index.html",
                result=traffic_level
            )

    return render_template("index.html")


def analyze_video(path):
    cap = cv2.VideoCapture(path)

    vehicle_count = 0
    frame_limit = 5   # keep very small for Render
    processed = 0

    while processed < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize (low memory)
        frame = cv2.resize(frame, (224, 224))

        # YOLO detection (light)
        results = model(frame, imgsz=224, conf=0.3)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # vehicle classes
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
