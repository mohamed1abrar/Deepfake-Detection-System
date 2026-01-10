import cv2
import numpy as np
import torch

def preprocess_xception(frame):
    img = cv2.resize(frame, (299, 299))
    img = img.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img).unsqueeze(0)
    return tensor
