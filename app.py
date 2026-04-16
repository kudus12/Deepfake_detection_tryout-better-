import os
import uuid
from flask import Flask, render_template, request, send_from_directory
from openpyxl import Workbook, load_workbook

from ML.inference import load_my_model, predict_video

app = Flask(__name__)

# ----------------------------
# 1) Upload folder
# ----------------------------
UPLOAD_FOLDER = "uploads"
DEBUG_FRAMES_FOLDER = "debug_frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FRAMES_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# 2) Load model once
# ----------------------------
MODEL_PATH = r"C:\Users\kudus\Desktop\try out\saved_models\best_model_tryout.pth"
my_model = load_my_model(MODEL_PATH)

# ----------------------------
# 3) Excel results file
# ----------------------------
EXCEL_PATH = r"C:\Users\kudus\Desktop\try out\test_results\tryout_model_results.xlsx"
os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)

# ----------------------------
# 4) Home
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ----------------------------
# Helper: safe extension check
# ----------------------------
def allowed_video(filename: str) -> bool:
    filename = filename.lower()
    return (
        filename.endswith(".mp4")
        or filename.endswith(".mov")
        or filename.endswith(".avi")
        or filename.endswith(".mkv")
    )


# ----------------------------
# Serve uploaded videos
# ----------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ----------------------------
# Serve saved debug frames
# ----------------------------
@app.route("/debug_frames/<path:filename>")
def debug_file(filename):
    return send_from_directory(DEBUG_FRAMES_FOLDER, filename)


# ----------------------------
# 5) Predict
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_video(file.filename):
        return render_template("index.html", error="Please upload a video file.")

    # Save with a unique name
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    # Make a unique debug folder per upload
    per_video_debug = os.path.join(DEBUG_FRAMES_FOLDER, os.path.splitext(unique_name)[0])
    os.makedirs(per_video_debug, exist_ok=True)

    # Prediction
    label, confidence, probs = predict_video(
        my_model,
        save_path,
        frame_limit=10
    )

    if label == "ERROR":
        return render_template("index.html", error="Could not read this video.")

    # Collect saved frame file names
    frame_files = []
    if os.path.exists(per_video_debug):
        frame_files = sorted(
            [f for f in os.listdir(per_video_debug) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    # ----------------------------
    # Save result to Excel
    # ----------------------------
    original_filename = file.filename
    filename_upper = original_filename.upper()

    if "FAKE" in filename_upper:
        actual_label = "FAKE"
    elif "REAL" in filename_upper:
        actual_label = "REAL"
    else:
        actual_label = "UNKNOWN"

    if actual_label == "UNKNOWN":
        result_status = "Unknown"
    else:
        result_status = "Correct" if label.upper() == actual_label else "Wrong"

    if not os.path.exists(EXCEL_PATH):
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append([
            "video_name",
            "actual_label",
            "predicted_label",
            "confidence",
            "real_probability",
            "fake_probability",
            "result_status"
        ])
        wb.save(EXCEL_PATH)

    wb = load_workbook(EXCEL_PATH)
    ws = wb.active

    ws.append([
        original_filename,
        actual_label,
        label.upper(),
        round(confidence, 2),
        round(probs[0] * 100, 2),
        round(probs[1] * 100, 2),
        result_status
    ])

    wb.save(EXCEL_PATH)

    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{confidence:.2f}",
        prob_real=f"{probs[0] * 100:.2f}",
        prob_fake=f"{probs[1] * 100:.2f}",
        uploaded_video=unique_name,
        debug_folder=os.path.splitext(unique_name)[0],
        frame_files=frame_files
    )


if __name__ == "__main__":
    app.run(debug=True)