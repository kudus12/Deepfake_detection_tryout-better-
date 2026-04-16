# ML/train_loop.py
# Try-out training:
# REAL = original
# FAKE = Deepfakes + FaceSwap + Face2Face

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset_loader import DeepfakeVideoDataset
from model import SimpleCNN

# Current try-out folder layout
REAL_DIR = r"C:\Users\kudus\Desktop\try out\Dataset\real\original"
DEEPFAKES_DIR = r"C:\Users\kudus\Desktop\try out\Dataset\fake\Deepfakes"
FACESWAP_DIR = r"C:\Users\kudus\Desktop\try out\FaceSwap"
FACE2FACE_DIR = r"C:\Users\kudus\Desktop\try out\Face2Face"


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = DeepfakeVideoDataset(
        real_dir=REAL_DIR,
        deepfakes_dir=DEEPFAKES_DIR,
        faceswap_dir=FACESWAP_DIR,
        face2face_dir=FACE2FACE_DIR,
        frame_limit=10,
        face_size=224,
        scan_frames=20,
        save_keyframes=False,
        keyframe_dir=r"C:\Users\kudus\Desktop\try out\saved_keyframes_tryout"
    )

    print("Total videos:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=2).to(device)
    print("Model ready ✅")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    epochs = 8
    best_val_acc = 0.0

    save_dir = r"C:\Users\kudus\Desktop\try out\saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_tryout_multifake.pth")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # TRAIN
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            c, t = accuracy_from_logits(logits, labels)
            train_correct += c
            train_total += t

        train_loss = train_loss_sum / train_total
        train_acc = (train_correct / train_total) * 100

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)

                logits = model(frames)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                c, t = accuracy_from_logits(logits, labels)
                val_correct += c
                val_total += t

        val_loss = val_loss_sum / val_total
        val_acc = (val_correct / val_total) * 100

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model: {save_path} (Val Acc {best_val_acc:.2f}%)")

    print("\nDone ✅")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Saved try-out model at:", save_path)


if __name__ == "__main__":
    main()