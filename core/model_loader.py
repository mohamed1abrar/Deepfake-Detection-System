import os
import torch
from network.models import model_selection

DEVICE = torch.device("cpu")

PRETRAINED_DIR = "pretrained_model"

MODEL_FILES = [
    "model.pth",
    "model_v0.pth",
    "model_v2.pth",
    "model_v3.pth"
]

def load_models():
    models = []

    for fname in MODEL_FILES:
        model_path = os.path.join(PRETRAINED_DIR, fname)
        if not os.path.exists(model_path):
            continue

        model = model_selection(
            modelname="xception",
            num_out_classes=2,
            dropout=0.5
        )

        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()

        models.append((fname.replace(".pth", ""), model))

    return models
