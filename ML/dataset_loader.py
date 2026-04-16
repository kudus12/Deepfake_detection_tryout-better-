# ML/dataset_loader.py
# Try-out dataset loader:
# REAL = original
# FAKE = Deepfakes + FaceSwap + Face2Face

import os
import cv2
import torch
from torch.utils.data import Dataset


def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class DeepfakeVideoDataset(Dataset):
    def __init__(
        self,
        real_dir,
        deepfakes_dir,
        faceswap_dir,
        face2face_dir,
        frame_limit=10,
        face_size=224,
        scan_frames=20,
        save_keyframes=False,
        keyframe_dir="saved_keyframes_tryout"
    ):
        self.samples = []
        self.frame_limit = frame_limit
        self.face_size = face_size
        self.scan_frames = scan_frames
        self.save_keyframes = save_keyframes
        self.keyframe_dir = keyframe_dir

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self.save_keyframes:
            os.makedirs(self.keyframe_dir, exist_ok=True)

        # REAL videos
        for file in os.listdir(real_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(real_dir, file), 0))

        # FAKE videos - Deepfakes
        for file in os.listdir(deepfakes_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(deepfakes_dir, file), 1))

        # FAKE videos - FaceSwap
        for file in os.listdir(faceswap_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(faceswap_dir, file), 1))

        # FAKE videos - Face2Face
        for file in os.listdir(face2face_dir):
            if file.lower().endswith(".mp4"):
                self.samples.append((os.path.join(face2face_dir, file), 1))

    def __len__(self):
        return len(self.samples)

    def _detect_face_and_eyes(self, frame):
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        face_found = False
        eyes_count = 0
        face_crop = None

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_found = True

            # face box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_crop = frame[y:y + h, x:x + w]

            face_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_detector.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5
            )
            eyes_count = len(eyes)

            # eye boxes
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(
                    annotated,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 0, 0),
                    2
                )
        else:
            # fallback center crop
            h, w, _ = frame.shape
            s = min(h, w)
            cx, cy = w // 2, h // 2
            x1 = max(cx - s // 2, 0)
            y1 = max(cy - s // 2, 0)
            face_crop = frame[y1:y1 + s, x1:x1 + s]

        # softer scoring
        score = frame_sharpness(frame)
        if face_found:
            score += 20
        if eyes_count >= 2:
            score += 10
        elif eyes_count == 1:
            score += 5

        return annotated, face_crop, score

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        cap = cv2.VideoCapture(video_path)

        candidates = []
        scanned = 0

        while cap.isOpened() and scanned < self.scan_frames:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, face_crop, score = self._detect_face_and_eyes(frame)

            candidates.append({
                "score": score,
                "annotated_frame": annotated,
                "face_crop": face_crop
            })

            scanned += 1

        cap.release()

        if len(candidates) == 0:
            return self.__getitem__((index + 1) % len(self.samples))

        # best frames first
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:self.frame_limit]

        frames = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        class_name = "real" if label == 0 else "fake"

        for i, item in enumerate(selected, start=1):
            face = item["face_crop"]

            if self.save_keyframes:
                out_dir = os.path.join(self.keyframe_dir, class_name, base_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"keyframe_{i}.jpg")
                cv2.imwrite(out_path, item["annotated_frame"])

            face = cv2.resize(face, (self.face_size, self.face_size))
            face = face[:, :, ::-1]  # BGR -> RGB
            face = face.astype("float32") / 255.0
            face = torch.from_numpy(face).permute(2, 0, 1)

            frames.append(face)

        while len(frames) < self.frame_limit:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames[:self.frame_limit])  # (T,C,H,W)
        return frames, label
