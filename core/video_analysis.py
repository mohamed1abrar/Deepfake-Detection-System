import cv2
import torch
import numpy as np
from .preprocessing import preprocess_xception

@torch.no_grad()
def analyze_video(video_path, models, max_frames=300):
    cap = cv2.VideoCapture(video_path)

    frame_scores = []
    frames_used = 0

    while frames_used < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_preds = []
        for _, model in models:
            inp = preprocess_xception(frame)
            out = model(inp)
            prob = torch.softmax(out, dim=1)[0][1].item()
            frame_preds.append(prob)

        frame_scores.append(np.mean(frame_preds))
        frames_used += 1

    cap.release()

    avg_fake = float(np.mean(frame_scores)) if frame_scores else 0.0
    label = "FAKE" if avg_fake >= 0.5 else "REAL"

    return {
        "is_image": False,
        "frames_used": frames_used,
        "ensemble": {
            "avg_fake": round(avg_fake, 4),
            "label": label
        }
    }
