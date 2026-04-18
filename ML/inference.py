import os
import cv2
import torch
import torch.nn.functional as F

from ML.model import SimpleCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_my_model(model_path):
    model = SimpleCNN(num_classes=2).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def detect_face_and_eyes(frame):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    face_found = False
    eyes_count = 0
    face_crop = None

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_found = True

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_crop = frame[y:y + h, x:x + w]

        face_gray = gray[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
        eyes_count = len(eyes)

        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(
                annotated,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0),
                2
            )
    else:
        h, w, _ = frame.shape
        s = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(cx - s // 2, 0)
        y1 = max(cy - s // 2, 0)
        face_crop = frame[y1:y1 + s, x1:x1 + s]

    score = frame_sharpness(frame)
    if face_found:
        score += 100.0
    if eyes_count >= 2:
        score += 50.0
    elif eyes_count == 1:
        score += 20.0

    return annotated, face_crop, score


def video_to_tensor(
    video_path,
    frame_limit=10,
    face_size=224,
    scan_frames=40,
    save_keyframes=True,
    keyframe_dir=None
):
    cap = cv2.VideoCapture(video_path)

    candidates = []
    scanned = 0

    while cap.isOpened() and scanned < scan_frames:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, face_crop, score = detect_face_and_eyes(frame)

        candidates.append({
            "score": score,
            "annotated_frame": annotated,
            "face_crop": face_crop
        })

        scanned += 1

    cap.release()

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[:frame_limit]

    frames = []

    if save_keyframes and keyframe_dir is not None:
        os.makedirs(keyframe_dir, exist_ok=True)

        # clear old images in that folder
        for old_file in os.listdir(keyframe_dir):
            if old_file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    os.remove(os.path.join(keyframe_dir, old_file))
                except Exception:
                    pass

    for i, item in enumerate(selected, start=1):
        face = item["face_crop"]

        if face is None or face.size == 0:
            continue

        if save_keyframes and keyframe_dir is not None:
            out_path = os.path.join(keyframe_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(out_path, item["annotated_frame"])

        face = cv2.resize(face, (face_size, face_size))
        face = face[:, :, ::-1]  # BGR -> RGB
        face = face.astype("float32") / 255.0
        face = torch.from_numpy(face).permute(2, 0, 1)
        frames.append(face)

    if len(frames) == 0:
        return None

    while len(frames) < frame_limit:
        frames.append(frames[-1].clone())

    frames = torch.stack(frames[:frame_limit])
    frames = frames.unsqueeze(0)
    return frames


def predict_video(model, video_path, frame_limit=10, debug_dir=None):
    frames = video_to_tensor(
        video_path,
        frame_limit=frame_limit,
        face_size=224,
        scan_frames=40,
        save_keyframes=True,
        keyframe_dir=debug_dir
    )

    if frames is None:
        return "ERROR", 0.0, [0.0, 0.0]

    frames = frames.to(DEVICE)

    with torch.no_grad():
        logits = model(frames)
        probs = F.softmax(logits, dim=1)

    prob_real = float(probs[0, 0].item())
    prob_fake = float(probs[0, 1].item())

    pred_idx = int(torch.argmax(probs, dim=1).item())

    if pred_idx == 0:
        return "REAL", prob_real * 100, [prob_real, prob_fake]
    else:
        return "FAKE", prob_fake * 100, [prob_real, prob_fake]