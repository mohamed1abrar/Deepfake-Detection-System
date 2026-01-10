import cv2
import torch
import numpy as np
from .preprocessing import preprocess_xception

@torch.no_grad()
def analyze_image(image_path, models):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("Invalid image file")

    model_preds = []

    for name, model in models:
        inp = preprocess_xception(image)
        out = model(inp)
        prob = torch.softmax(out, dim=1)[0][1].item()
        model_preds.append((name, prob))

    avg_fake = float(np.mean([p for _, p in model_preds]))
    label = "FAKE" if avg_fake >= 0.5 else "REAL"

    return {
        "is_image": True,
        "ensemble": {
            "avg_fake": round(avg_fake, 4),
            "label": label
        },
        "per_model": [
            {
                "name": name,
                "fake_prob": round(prob, 4)
            }
            for name, prob in model_preds
        ]
    }
