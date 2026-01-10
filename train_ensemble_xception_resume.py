# train_ensemble_xception_resume.py
# Resumable ensemble training for Xception-compatible models (Windows-friendly).
# Usage example (CMD):
# python train_ensemble_xception_resume.py --data_root "D:/TRAIN_IMAGES_500" --out_dir "./pretrained_model" --n_models 4 --epochs 12 --batch 24 --workers 6 --img_size 299 --amp

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# import your Xception factory (must be in PYTHONPATH / project root)
try:
    from network.models import model_selection
except Exception as e:
    print("[ERROR] Could not import network.models.model_selection:", e)
    raise

# --------------------------
# Top-level Dataset wrapper so Windows multiprocessing can pickle it
# --------------------------
from torch.utils.data import Dataset

class RemappedSubset(Dataset):
    """
    Wrapper around a Subset that remaps labels from original ImageFolder
    mapping to 'real'->0, everything else -> 1 (fake).
    Implemented at top-level so multiprocessing can pickle it on Windows.
    """
    def __init__(self, subset, idx_to_class):
        self.subset = subset
        self.idx_to_class = idx_to_class  # dict: orig_idx -> class_name

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, lab = self.subset[i]
        # lab is the original integer label from ImageFolder
        # map original index -> classname then remap to 0/1
        cname = self.idx_to_class[int(lab)].lower() if lab is not None else "fake"
        new_lab = 0 if cname == "real" else 1
        return img, new_lab

# --------------------------
# Utility functions
# --------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_transforms(img_size=299):
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
    ])
    val_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_t, val_t

def build_xception(device, dropout=0.5):
    m = model_selection(modelname="xception", num_out_classes=2, dropout=dropout)
    m = m.to(device)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
        print(f"[INFO] DataParallel on {torch.cuda.device_count()} GPUs")
    return m

def metrics_from_logits(logits_arr, targets):
    probs = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets, probs)
    except Exception:
        auc = float('nan')
    return {"acc":acc,"prec":prec,"rec":rec,"f1":f1,"auc":auc,"probs":probs,"preds":preds}

def save_splits(out_dir, indices):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "splits.npz"), **indices)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    losses = []
    all_logits = []
    all_targets = []
    pbar = tqdm(loader, desc="train_batches", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
    all_logits = np.vstack(all_logits) if len(all_logits) else np.zeros((0,2))
    return np.mean(losses) if losses else 0.0, all_logits, np.array(all_targets)

def eval_loader(model, loader, criterion, device):
    model.eval()
    losses = []
    all_logits = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="val_batches", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    all_logits = np.vstack(all_logits) if len(all_logits) else np.zeros((0,2))
    return np.mean(losses) if losses else 0.0, all_logits, np.array(all_targets)

def choose_save_name(out_dir, idx, total):
    if total == 4:
        names = ["model.pth", "model_v0.pth", "model_v2.pth", "model_v3.pth"]
        return os.path.join(out_dir, names[idx])
    else:
        return os.path.join(out_dir, f"model_ensemble_{idx}.pth")

def safe_load_checkpoint(path, model, optimizer=None, device='cpu'):
    ck = torch.load(path, map_location=device)
    # support either state-dict dict or {'state_dict':...}
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    # strip "module." if present
    new_state = {k.replace("module.", ""): v for k,v in state.items()}
    model.load_state_dict(new_state, strict=False)
    if optimizer is not None and isinstance(ck, dict) and "optimizer" in ck:
        try:
            optimizer.load_state_dict(ck["optimizer"])
        except Exception as e:
            print("[WARN] Could not load optimizer state:", e)
    return ck  # return full checkpoint dict if available

# --------------------------
# Main
# --------------------------
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("[INFO] Device:", device)

    train_tf, val_tf = make_transforms(img_size=args.img_size)
    dataset = datasets.ImageFolder(args.data_root, transform=train_tf)
    if len(dataset) == 0:
        raise RuntimeError("No images found in data_root: " + args.data_root)
    print("[INFO] Found classes:", dataset.classes, "mapping:", dataset.class_to_idx)

    # deterministic split
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise RuntimeError("Not enough images after split")
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train+n_val].tolist()
    test_idx = indices[n_train+n_val:].tolist()
    save_splits(args.out_dir, {"train": np.array(train_idx), "val": np.array(val_idx), "test": np.array(test_idx)})
    print(f"[INFO] splits saved to {args.out_dir}/splits.npz (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    # remap labels and create RemappedSubset wrappers (top-level class)
    idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}
    train_subset = RemappedSubset(Subset(dataset, train_idx), idx_to_class)
    val_subset = RemappedSubset(Subset(dataset, val_idx), idx_to_class)

    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Outer loop over models (with progress)
    models_iter = range(args.n_models)
    models_pbar = tqdm(models_iter, desc="models", unit="model")
    for m_idx in models_pbar:
        model_seed = args.seed + m_idx*101
        seed_everything(model_seed)
        models_pbar.set_postfix({"model_idx": m_idx})
        # build model
        model = build_xception(device, dropout=args.dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # NOTE: removed verbose arg for compatibility with older torch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and args.amp) else None

        # determine save path
        save_path = choose_save_name(args.out_dir, m_idx, args.n_models)
        start_epoch = 1
        best_f1 = -1.0

        # resume if file exists and resume flag on
        if args.resume_existing and os.path.exists(save_path) and not args.force_reinit:
            try:
                ck = safe_load_checkpoint(save_path, model, optimizer=optimizer, device=device)
                prev_epoch = int(ck.get("epoch", 0))
                best_f1 = float(ck.get("f1", best_f1))
                start_epoch = prev_epoch + 1
                print(f"[Model {m_idx}] Resuming from {save_path}, epoch -> {start_epoch}, best_f1 -> {best_f1:.4f}")
            except Exception as e:
                print(f"[WARN] Could not resume {save_path}: {e}. Training from scratch.")
                start_epoch = 1
                best_f1 = -1.0
        else:
            if os.path.exists(save_path) and args.force_reinit:
                print(f"[Model {m_idx}] --force_reinit set; ignoring existing checkpoint {save_path} and reinitializing.")
            else:
                print(f"[Model {m_idx}] No checkpoint found at {save_path}, starting from scratch.")

        # epochs loop with tqdm (shows epoch progress for this model)
        for epoch in range(start_epoch, args.epochs+1):
            epoch_pbar = tqdm(total=1, desc=f"model{m_idx}_epoch{epoch}", leave=False)
            print(f"[Model {m_idx}] Epoch {epoch}/{args.epochs}")
            tr_loss, tr_logits, tr_targets = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
            tr_metrics = metrics_from_logits(tr_logits, tr_targets)

            val_loss, val_logits, val_targets = eval_loader(model, val_loader, criterion, device)
            val_metrics = metrics_from_logits(val_logits, val_targets)

            print(f" Train loss: {tr_loss:.4f} acc:{tr_metrics['acc']:.3f} f1:{tr_metrics['f1']:.3f}")
            print(f" Val  loss: {val_loss:.4f} acc:{val_metrics['acc']:.3f} f1:{val_metrics['f1']:.3f}")

            scheduler.step(val_metrics['f1'])

            state_dict_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            ckpt = {
                "epoch": epoch,
                "model_name": "xception",
                "state_dict": state_dict_to_save,
                "optimizer": optimizer.state_dict(),
                "f1": float(val_metrics['f1'])
            }

            # save best and periodic
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(ckpt, save_path)
                print(f"[Model {m_idx}] Saved new best -> {save_path} (f1={best_f1:.4f})")
            if epoch % args.save_every == 0:
                epath = os.path.join(args.out_dir, f"model_m{m_idx}_epoch{epoch}.pth")
                torch.save(ckpt, epath)
                print(f"[Model {m_idx}] saved periodic checkpoint: {epath}")

            epoch_pbar.update(1)
            epoch_pbar.close()

        print(f"[Model {m_idx}] Finished. Best F1: {best_f1:.4f}")
        models_pbar.update()

    print("[ALL MODELS DONE]")

# --------------------------
# CLI & entrypoint
# --------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows: helps child processes spawn correctly

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="./pretrained_model")
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_models", type=int, default=4, help="how many models to train for ensemble")
    parser.add_argument("--resume_existing", action="store_true", help="if set, resume training from existing checkpoint files (default ON)")
    parser.add_argument("--force_reinit", action="store_true", help="if set, ignore existing checkpoints and reinitialize models")
    args = parser.parse_args()

    # default behavior: resume_existing True unless user explicitly passes --force_reinit
    if not ('--resume_existing' in " ".join(os.sys.argv)) and not args.force_reinit:
        args.resume_existing = True

    main(args)
