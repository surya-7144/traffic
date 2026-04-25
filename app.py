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

    # 🔥 ONLY PROCESS 2 FRAMES (VERY LIGHT)
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 VERY SMALL SIZE
        frame = cv2.resize(frame, (160, 160))

        # 🔥 FAST YOLO
        results = model(frame, imgsz=160, conf=0.4)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    vehicle_count += 1

    cap.release()

    if vehicle_count < 3:
        return "Low"
    elif vehicle_count < 8:
        return "Medium"
    else:
        return "High"

if __name__ == "__main__":
    app.run(debug=True)
