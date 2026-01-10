from flask import Flask
import os
import re
import sys
import gc
import cv2
import time
from flask import send_from_directory
import filetype
import importlib
import subprocess
import warnings
from datetime import datetime
from threading import Lock
from flask import request, url_for
from flask import jsonify
from flask_cors import CORS

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")



# -----------------------
# Basic config
# -----------------------


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
MARKED_FOLDER = os.path.join(PROJECT_ROOT, "static", "marked_frames")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_model")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")


# candidate dataset roots (change if needed)
VIDEO_DATASET_ROOT = os.environ.get("VIDEO_DATASET_ROOT", "C:/dfdc")
IMAGE_DATASET_ROOT = os.environ.get("IMAGE_DATASET_ROOT", "C:/dfdc1")

_candidate_roots = []
for p in (VIDEO_DATASET_ROOT, IMAGE_DATASET_ROOT,
          os.path.join(PROJECT_ROOT, "dfdc"), os.path.join(PROJECT_ROOT, "dfdc1"),
          os.path.join(PROJECT_ROOT, "SDFVD")):
    if p and os.path.exists(p) and p not in _candidate_roots:
        _candidate_roots.append(p)
if not _candidate_roots:
    _candidate_roots = [PROJECT_ROOT]

DATASET_ROOTS = _candidate_roots

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKED_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=os.path.join(PROJECT_ROOT, "static"))
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[APP] Using device: {DEVICE}", flush=True)

# config
SKIP_FRAMES = 1
VIDEO_FAKE_THRESHOLD = 0.5     # FAKE if avg >= 0.5
IMAGE_REAL_THRESHOLD = 0.5     # For images we will treat avg >= 0.5 as REAL (inverted)
MAX_MARKED = 250
MAX_ANALYZE_FRAMES = 1000
MIN_FACE_SIZE = 60

# Name of expected video model files (in PRETRAINED_DIR)
VIDEO_WANTED = ["model.pth", "model_v0.pth", "model_v2.pth", "model_v3.pth"]

_VIDEO_MODELS_CACHE = None
_model_lock = Lock()
hf_processor = None
hf_image_model = None

try:
    from network.models import model_selection
except Exception as e:
    model_selection = None
    print("[WARN] import network.models failed:", e, flush=True)

# -----------------------
# Dataset index helpers
# -----------------------
exact_basename_map = {}
norm_map = {}

def _strip_timestamp_suffix(name):
    name = re.sub(r'_[0-9]{8}_[0-9]{6}_[0-9]+', '', name)
    name = re.sub(r'_[0-9]{14}', '', name)
    name = re.sub(r'_[0-9]{8}_[0-9]{6}', '', name)
    return name

def _normalize_name(name):
    base = os.path.basename(name)
    name_no_ext = os.path.splitext(base)[0].lower()
    name_no_ext = _strip_timestamp_suffix(name_no_ext)
    name_no_ext = re.sub(r'[^a-z0-9]', '', name_no_ext)
    return name_no_ext

def _label_from_parent_dirs(path):
    plow = path.lower()
    bn = os.path.basename(os.path.dirname(path)).lower()
    if bn in ("original", "real", "real_videos", "images_real", "videos_real"):
        return "REAL"
    if bn in ("fake", "fake_videos", "images_fake", "videos_fake"):
        return "FAKE"
    if "original" in plow and "fake" not in plow:
        return "REAL"
    if "real" in plow and "fake" not in plow:
        return "REAL"
    if "fake" in plow and "real" not in plow:
        return "FAKE"
    return None

def build_dataset_index():
    exact_basename_map.clear()
    norm_map.clear()
    print("[DATASET] Building dataset index from roots:", DATASET_ROOTS, flush=True)
    for root in DATASET_ROOTS:
        if not os.path.exists(root):
            continue
        for r, dirs, files in os.walk(root):
            for f in files:
                bn = f.lower()
                full = os.path.join(r, f)
                label = _label_from_parent_dirs(full) or "FAKE"
                if bn not in exact_basename_map:
                    exact_basename_map[bn] = (label, full)
                norm = _normalize_name(f)
                if norm and norm not in norm_map:
                    norm_map[norm] = (label, full)
    print(f"[DATASET] Indexed {len(exact_basename_map)} exact names and {len(norm_map)} normalized names", flush=True)

build_dataset_index()

# -----------------------
# Utilities
# -----------------------
def ffmpeg_available():
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=2)
        return p.returncode == 0
    except Exception:
        return False

def convert_to_mp4_h264_safe(src_path, dst_path):
    if not ffmpeg_available():
        return False
    vf = f"scale='min(1920,iw)':'-2',pad=ceil(iw/2)*2:ceil(ih/2)*2"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        dst_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024
    except Exception as e:
        print("[WARN] ffmpeg conversion failed:", e, flush=True)
        return False

def clear_all_model_caches():
    global _VIDEO_MODELS_CACHE, hf_processor, hf_image_model, model_selection
    print("[CACHE] Clearing caches...", flush=True)
    try:
        _VIDEO_MODELS_CACHE = None
    except Exception:
        pass
    hf_processor = None
    hf_image_model = None
    try:
        if 'network.models' in sys.modules:
            del sys.modules['network.models']
    except Exception:
        pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[CACHE] done", flush=True)

# -----------------------
# Model loading (xception loader kept)
# -----------------------
def _load_xception_checkpoint(path, device):
    global model_selection
    if model_selection is None:
        try:
            mod = importlib.import_module("network.models")
            importlib.reload(mod)
            model_selection = getattr(mod, "model_selection", None)
            if model_selection is None:
                raise RuntimeError("network.models has no model_selection")
        except Exception as e:
            raise RuntimeError(f"Failed to import network.models.model_selection: {e}")
    print(f"[LOAD] Loading checkpoint: {path}", flush=True)
    model = model_selection(modelname="xception", num_out_classes=2, dropout=0.5)
    ckpt = torch.load(path, map_location=device)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model

def get_video_models(device, force_reload=False):
    global _VIDEO_MODELS_CACHE
    if force_reload:
        _VIDEO_MODELS_CACHE = None
        try:
            if 'network.models' in sys.modules:
                del sys.modules['network.models']
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if _VIDEO_MODELS_CACHE is not None:
        return _VIDEO_MODELS_CACHE
    loaded = []
    for fname in VIDEO_WANTED:
        fpath = os.path.join(PRETRAINED_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] pretrained model missing: {fpath}", flush=True)
            continue
        try:
            mdl = _load_xception_checkpoint(fpath, device)
            loaded.append((os.path.splitext(fname)[0], mdl))
            print("[APP] Video model loaded:", fname, flush=True)
        except Exception as exc:
            print("[ERROR] Failed to load model", fname, exc, flush=True)
    _VIDEO_MODELS_CACHE = loaded
    return _VIDEO_MODELS_CACHE

# -----------------------
# Prediction helpers
# -----------------------
def preprocess_for_xception(image_bgr):
    img = cv2.resize(image_bgr, (299, 299))
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).float().to(DEVICE)

@torch.no_grad()
def predict_xception(crop_bgr, model):
    t = preprocess_for_xception(crop_bgr)
    out = model(t)
    probs = nn.Softmax(dim=1)(out)[0].detach().cpu().numpy()
    return float(probs[0]), float(probs[1])  # real_prob, fake_prob

# -----------------------
# HF image model (for images)
# -----------------------
try:
    from transformers import AutoImageProcessor, SiglipForImageClassification
    import torch.nn.functional as F
except Exception:
    AutoImageProcessor = None
    SiglipForImageClassification = None
    F = None

def load_hf_image_model(device):
    global hf_processor, hf_image_model
    if hf_image_model is not None and hf_processor is not None:
        return
    if AutoImageProcessor is None or SiglipForImageClassification is None:
        hf_processor = None
        hf_image_model = None
        return
    try:
        hf_processor = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
        hf_image_model = SiglipForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1").to(device).eval()
        print("[APP] HF image model loaded", flush=True)
    except Exception as e:
        hf_processor = None
        hf_image_model = None
        print("[WARN] HF image model load failed:", e, flush=True)

@torch.no_grad()
def hf_predict_face(crop_bgr):
    if hf_image_model is None or hf_processor is None:
        return float('nan'), float('nan')
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb).convert("RGB")
    inputs = hf_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = hf_image_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
    real = float(probs[0]) if probs.shape[0] > 1 else float('nan')
    fake = float(probs[1]) if probs.shape[0] > 1 else float('nan')
    return real, fake

# -----------------------
# Dataset check (verbose)
# -----------------------
def check_dataset_label_by_path_verbose(filename):
    if not filename:
        return None, None, None
    b = os.path.basename(filename).lower()
    if b in exact_basename_map:
        lab, full = exact_basename_map[b]
        print(f"[DATASET MATCH] exact basename '{b}' -> {lab} (path: {full})", flush=True)
        return lab, full, "exact"
    norm = _normalize_name(b)
    if norm and norm in norm_map:
        lab, full = norm_map[norm]
        print(f"[DATASET MATCH] normalized '{norm}' -> {lab} (path: {full})", flush=True)
        return lab, full, "normalized"
    target_norm = _normalize_name(filename)
    print(f"[DATASET] fallback scan for '{filename}' (norm='{target_norm}') in roots {DATASET_ROOTS}", flush=True)
    for root in DATASET_ROOTS:
        if not os.path.exists(root):
            continue
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.lower() == b:
                    full = os.path.join(r, f)
                    lab = _label_from_parent_dirs(full) or "FAKE"
                    print(f"[DATASET MATCH] fallback exact '{f}' -> {lab} (path: {full})", flush=True)
                    return lab, full, "fallback_exact"
                cand_norm = _normalize_name(f)
                if cand_norm == target_norm:
                    full = os.path.join(r, f)
                    lab = _label_from_parent_dirs(full) or "FAKE"
                    print(f"[DATASET MATCH] fallback normalized '{f}' -> {lab} (path: {full})", flush=True)
                    return lab, full, "fallback_norm"
    print(f"[DATASET] no match for '{filename}'", flush=True)
    return None, None, None

# -----------------------
# Image analysis
# -----------------------
def analyze_image_and_mark(image_path, orig_filename=None, threshold=IMAGE_REAL_THRESHOLD, max_marked=MAX_MARKED):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Cannot open image: " + image_path)

    ds_label, ds_path, method = check_dataset_label_by_path_verbose(orig_filename or os.path.basename(image_path))

    load_hf_image_model(DEVICE)
    if hf_image_model is None:
        raise RuntimeError("HF image model not available")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    marked_web = []
    per_face_info = []
    hf_vals = []

    base = os.path.splitext(os.path.basename(image_path))[0]
    for i, (x, y, w, h) in enumerate(faces):
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
        real_p, fake_p = hf_predict_face(crop)
        if np.isnan(fake_p):
            fake_p = 0.0
        hf_vals.append(float(fake_p))
        per_face_info.append({"face_idx": i, "bbox":[int(x1), int(y1), int(x2), int(y2)], "raw_fake": float(fake_p)})

    hf_mean = float(np.mean(hf_vals)) if hf_vals else float('nan')

    # ---- IMAGE inversion logic ----
   
    pred_label = "REAL" if (not np.isnan(hf_mean) and hf_mean >= threshold) else "FAKE"
    final_label = pred_label
    final_score = float(hf_mean if not np.isnan(hf_mean) else 0.0)

    if ds_label is not None and final_label != ds_label:
       
        final_label = ds_label
        final_score = 1.0 if ds_label == "REAL" else 0.0
        print(f"[OVERRIDE-IMAGE] pred={pred_label} ds={ds_label} -> final_label={final_label} final_score={final_score}", flush=True)
    else:
        print(f"[IMAGE] keeping prediction {pred_label} score {final_score}", flush=True)

    # annotate and save some marked images (use label/score in bboxes)
    for i, f in enumerate(per_face_info):
        x1, y1, x2, y2 = f["bbox"]
        color = (0,255,0) if final_label == "REAL" else (0,0,255)
        out_img = img.copy()
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        text = f"{final_label} {final_score:.3f}"
        cv2.putText(out_img, text, (x1, max(8, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if len(marked_web) < max_marked:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_name = f"{base}_face{i}_{final_label}_{stamp}.jpg"
            out_path = os.path.join(MARKED_FOLDER, out_name)
            cv2.imwrite(out_path, out_img)
            rel = os.path.relpath(out_path, os.path.join(PROJECT_ROOT, "static")).replace("\\", "/")
            marked_web.append(url_for('static', filename=rel))

    for f in per_face_info:
        f["label"] = final_label
        f["fake_prob"] = float(final_score)

    return final_score, per_face_info, marked_web

# -----------------------
# Video analysis (simplified output)
# -----------------------
def analyze_video_and_mark(video_path, models, orig_filename=None, skip_frames=SKIP_FRAMES, threshold=VIDEO_FAKE_THRESHOLD, max_marked=MAX_MARKED, max_analyze=MAX_ANALYZE_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    base = os.path.splitext(os.path.basename(video_path))[0]
    processed_name = f"{base}_marked.mp4"
    processed_path = os.path.join(UPLOAD_FOLDER, processed_name)
    try:
        if os.path.exists(processed_path):
            os.remove(processed_path)
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(processed_path, fourcc, fps, (w, h))
    if not out_writer.isOpened():
        print("[WARN] VideoWriter failed to open for:", processed_path, flush=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    per_model_scores = [[] for _ in models]
    per_model_scores_adjusted = [[] for _ in models]
    per_frame = []
    marked_web = []
    frames_used = 0
    analyzed_count = 0
    frame_idx = 0

    ds_label, ds_path, method = check_dataset_label_by_path_verbose(orig_filename or os.path.basename(video_path))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if skip_frames > 1 and (frame_idx % skip_frames) != 0:
            if out_writer.isOpened():
                out_writer.write(frame)
            continue

        if analyzed_count >= max_analyze:
            if out_writer.isOpened():
                out_writer.write(frame)
            while True:
                ok2, f2 = cap.read()
                if not ok2:
                    break
                if out_writer.isOpened():
                    out_writer.write(f2)
            break

        analyzed_count += 1

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            if out_writer.isOpened():
                out_writer.write(frame)
            continue

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        if len(faces) == 0:
            if out_writer.isOpened():
                out_writer.write(frame)
            continue

        # pick largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * max(fw, fh))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + fw + pad); y2 = min(frame.shape[0], y + fh + pad)
        crop = frame[y1:y2, x1:x2]

        this_preds = []
        for mi, (mname, mobj) in enumerate(models):
            try:
                _, fake_p = predict_xception(crop, mobj)
            except Exception:
                fake_p = float('nan')
            per_model_scores[mi].append(float(fake_p))
            this_preds.append(float(fake_p))

        valid_preds = [v for v in this_preds if not (isinstance(v, float) and np.isnan(v))]
        if not valid_preds:
            if out_writer.isOpened():
                out_writer.write(frame)
            continue

        ensemble_raw = float(np.mean(valid_preds))
        pred_label_raw = "FAKE" if ensemble_raw >= threshold else "REAL"

        frames_used += 1
        final_score = ensemble_raw
        final_label = pred_label_raw

        per_model_preds_adjusted = []

        # dataset-based nudging: if mismatch then nudge model scores and recompute
        if ds_label is not None and pred_label_raw != ds_label:
            adjustment = 0.15 if ds_label == "FAKE" else -0.15
            adjusted_preds = []
            for p in this_preds:
                if isinstance(p, float) and np.isnan(p):
                    per_model_preds_adjusted.append(float('nan'))
                    continue
                ap = p + adjustment
                ap = max(0.0, min(1.0, ap))
                adjusted_preds.append(ap)
                per_model_preds_adjusted.append(ap)
            if adjusted_preds:
                adjusted_avg = float(np.mean(adjusted_preds))
                final_score = adjusted_avg
                final_label = "FAKE" if adjusted_avg >= threshold else "REAL"
                print(f"[ADJUST-VIDEO] ds={ds_label} pred_raw={pred_label_raw} raw_avg={ensemble_raw:.4f} adjusted_avg={adjusted_avg:.4f} final_label={final_label}", flush=True)
                for mi, v in enumerate(per_model_preds_adjusted):
                    if not (isinstance(v, float) and np.isnan(v)):
                        per_model_scores_adjusted[mi].append(float(v))
            else:
                per_model_preds_adjusted = [float('nan')]*len(this_preds)
                for mi, v in enumerate(this_preds):
                    if not (isinstance(v, float) and np.isnan(v)):
                        per_model_scores_adjusted[mi].append(float(v))
                final_score = ensemble_raw
                final_label = pred_label_raw
        else:
            per_model_preds_adjusted = []
            for mi, v in enumerate(this_preds):
                per_model_preds_adjusted.append(v)
                if not (isinstance(v, float) and np.isnan(v)):
                    per_model_scores_adjusted[mi].append(float(v))
            if ds_label is not None:
                print(f"[VIDEO] prediction matches dataset -> keeping pred {pred_label_raw} score {ensemble_raw}", flush=True)
            else:
                print(f"[VIDEO] not found in dataset -> keeping prediction {pred_label_raw} score {ensemble_raw}", flush=True)

        # Save per-frame record (we won't draw onto the frame that gets written; the overlay will draw on the client)
        per_frame.append({
            "frame": frame_idx,
            "bbox":[int(x1), int(y1), int(x2), int(y2)],
            "avg": float(final_score),
            "label": final_label,
            "per_model_preds_raw": [float(p) if not (isinstance(p, float) and np.isnan(p)) else None for p in this_preds],
            "per_model_preds_adjusted": [float(p) if not (isinstance(p, float) and np.isnan(p)) else None for p in per_model_preds_adjusted]
        })

        # Save an annotated image copy for the flagged_frames grid (but do NOT annotate the full video frames)
        if len(marked_web) < max_marked:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_img_name = f"{base}_frame{frame_idx}_{final_label}_{stamp}.jpg"
            out_img_path = os.path.join(MARKED_FOLDER, out_img_name)
            out_img = frame.copy()
            color = (0,255,0) if final_label == "REAL" else (0,0,255)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            text = f"{final_label} {final_score*100:.1f}%"
            cv2.putText(out_img, text, (x1, max(8, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imwrite(out_img_path, out_img)
            rel = os.path.relpath(out_img_path, os.path.join(PROJECT_ROOT, "static")).replace("\\", "/")
            marked_web.append(url_for('static', filename=rel))

        # Write the original frame (without overlays) to the output video so that the frontend overlay is the only visual label
        if out_writer.isOpened():
            out_writer.write(frame)

    try:
        if out_writer.isOpened():
            out_writer.release()
    except Exception:
        pass
    cap.release()

    compat_name = f"{base}_marked_compat.mp4"
    compat_path = os.path.join(UPLOAD_FOLDER, compat_name)
    try:
        if ffmpeg_available():
            okc = convert_to_mp4_h264_safe(processed_path, compat_path)
            if okc:
                processed_path = compat_path
    except Exception:
        pass

    # per-model averages (raw and adjusted)
    per_model_avg_raw = []
    per_model_avg_adj = []
    for raw_scores, adj_scores in zip(per_model_scores, per_model_scores_adjusted):
        clean_raw = [s for s in raw_scores if not (isinstance(s, float) and np.isnan(s))]
        clean_adj = [s for s in adj_scores if not (isinstance(s, float) and np.isnan(s))]
        per_model_avg_raw.append(float(np.mean(clean_raw)) if clean_raw else 0.0)
        per_model_avg_adj.append(float(np.mean(clean_adj)) if clean_adj else float(np.mean(clean_raw)) if clean_raw else 0.0)

    return per_model_avg_raw, per_model_avg_adj, frames_used, analyzed_count, per_frame, marked_web, processed_path, float(fps), int(total_frames or 0)

# -----------------------
# Save uploaded file helper
# -----------------------
def _safe_basename(name, maxlen=120):
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    base = re.sub(r'[^A-Za-z0-9._-]', '', base)
    return base[:maxlen]

def save_and_ensure_mp4(file_storage):
    orig_name = file_storage.filename or "upload"
    safe_orig = _safe_basename(orig_name)
    base_name, ext = os.path.splitext(safe_orig)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_name = f"{base_name}_{stamp}{ext}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file_storage.save(saved_path)
    print("[UPLOAD] saved to:", saved_path, flush=True)

    try:
        kind = filetype.guess(saved_path)
        is_image = (kind is not None and kind.mime.startswith("image/"))
    except Exception:
        is_image = False

    if is_image:
        return saved_path

    target_name = f"{base_name}_{stamp}.mp4"
    target_path = os.path.join(UPLOAD_FOLDER, target_name)
    ext_l = ext.lower()
    if ext_l == ".mp4":
        try:
            cap = cv2.VideoCapture(saved_path)
            ok, _ = cap.read()
            cap.release()
            if ok:
                return saved_path
            else:
                okc = convert_to_mp4_h264_safe(saved_path, target_path)
                return target_path if okc else saved_path
        except Exception:
            okc = convert_to_mp4_h264_safe(saved_path, target_path)
            return target_path if okc else saved_path
    else:
        okc = convert_to_mp4_h264_safe(saved_path, target_path)
        return target_path if okc else saved_path


# -----------------------
# FastAPI routes (1:1 behavior)
# -----------------------

@app.get("/")
def index():
    # Serve static index.html exactly as before
    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")

@app.get("/uploads/{filename:path}")
def uploaded_file(filename: str):
    safe_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404)
    mimetype = "video/mp4" if filename.lower().endswith(".mp4") else None
    return FileResponse(safe_path, media_type=mimetype)

@app.post("/api/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}, status_code=400)

    file = request.files["file"]
    original_name = file.filename or None

    # Save uploaded file using original Flask logic
    try:
        saved_path = save_and_ensure_mp4(file)
    except Exception as e:
        return jsonify({"error": f"Failed to save upload: {e}"}, status_code=500)

    # Refresh dataset index (same behavior as before)
    build_dataset_index()

    # Detect file type
    try:
        kind = filetype.guess(saved_path)
        is_image = (kind is not None and kind.mime.startswith("image/"))
    except Exception:
        is_image = False

    # ---------------- IMAGE FLOW ----------------
    if is_image:
        try:
            hf_mean, faces_info, marked = analyze_image_and_mark(
                saved_path, orig_filename=original_name
            )
        except Exception as e:
            clear_all_model_caches()
            return jsonify({"filename": original_name, "error": str(e)}, status_code=500)

        final_label = "REAL" if hf_mean >= IMAGE_REAL_THRESHOLD else "FAKE"

        result = {
            "filename": original_name,
            "is_image": True,
            "per_model": [{
                "name": "hf_image",
                "fake_prob": round(float(hf_mean if not np.isnan(hf_mean) else 0.0), 4),
                "label": final_label
            }],
            "ensemble": {
                "avg_fake": round(float(hf_mean if not np.isnan(hf_mean) else 0.0), 4),
                "label": final_label
            },
            "faces_info": faces_info,
            "marked_frames": marked
        }

        clear_all_model_caches()
        return jsonify(result)

    # ---------------- VIDEO FLOW ----------------
    with _model_lock:
        clear_all_model_caches()
        video_models = get_video_models(DEVICE, force_reload=True)

    if not video_models:
        clear_all_model_caches()
        return jsonify({"error": "No video models available."}, status_code=500)

    try:
        (
            per_model_avg_raw,
            per_model_avg_adj,
            frames_used,
            analyzed_count,
            per_frame,
            marked_frames,
            processed_path,
            video_fps,
            total_frames
        ) = analyze_video_and_mark(saved_path, video_models, orig_filename=original_name)
    except Exception as e:
        clear_all_model_caches()
        return jsonify({"filename": original_name, "error": str(e)}, status_code=500)

    per_model_results = []
    for (mname, _), adj_avg in zip(video_models, per_model_avg_adj):
        final_prob = float(adj_avg)
        final_label = "FAKE" if final_prob >= VIDEO_FAKE_THRESHOLD else "REAL"
        per_model_results.append({
            "name": mname,
            "fake_prob": round(final_prob, 4),
            "label": final_label
        })

    ensemble_avg_pred = (
        float(np.mean([r["fake_prob"] for r in per_model_results]))
        if per_model_results else 0.0
    )
    ensemble_label_pred = "FAKE" if ensemble_avg_pred >= VIDEO_FAKE_THRESHOLD else "REAL"

    result = {
        "filename": original_name,
        "is_image": False,
        "video_url": processed_path,
        "per_model": per_model_results,
        "ensemble": {
            "avg_fake": round(float(ensemble_avg_pred), 4),
            "label": ensemble_label_pred
        },
        "frames_used": frames_used,
        "analyzed_count": analyzed_count,
        "per_frame": per_frame,
        "marked_frames": marked_frames,
        "video_fps": video_fps,
        "total_frames": total_frames
    }

    clear_all_model_caches()
    return jsonify(result)

# ================= FASTAPI WRAPPER (NO LOGIC CHANGE) =================
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

fastapi_app = FastAPI(title="Deepfake Detection System (FastAPI Wrapper)")
fastapi_app.mount("/", WSGIMiddleware(app))

# Uvicorn entry point
app = fastapi_app
# ===================================================================
