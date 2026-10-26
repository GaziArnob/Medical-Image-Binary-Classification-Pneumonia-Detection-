"""Shared utilities: seeding, data loading, models, training loop, metrics."""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PneumoniaMNIST
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- data
def load_split(split: str, size: int = 28):
    """Return (uint8 images [N,H,W], int labels [N]) for the supplied split.
    Label 0 = normal, 1 = pneumonia (positive class)."""
    ds = PneumoniaMNIST(split=split, download=True, root=str(DATA_DIR), size=size)
    return ds.imgs.astype(np.uint8), ds.labels.ravel().astype(np.int64)


class XrayDataset(Dataset):
    """Wraps uint8 images; applies optional GPU-free tensor augmentation
    (training split only) and optional resize / grayscale->RGB conversion."""

    def __init__(self, imgs, labels, augment=False, out_size=None, rgb=False,
                 mean=0.5, std=0.5):
        self.x = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(1)  # [N,1,H,W]
        self.y = torch.from_numpy(labels)
        self.augment, self.out_size, self.rgb = augment, out_size, rgb
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.y)

    def _augment(self, x):
        # Mild, anatomically plausible: small rotation/translation/scale + brightness/contrast.
        # No horizontal flip (heart position / lateral markers carry meaning).
        angle = np.deg2rad(random.uniform(-10, 10))
        scale = random.uniform(0.9, 1.1)
        tx, ty = random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08)
        c, s = np.cos(angle) / scale, np.sin(angle) / scale
        theta = torch.tensor([[c, -s, tx], [s, c, ty]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
        x = F.grid_sample(x.unsqueeze(0), grid, padding_mode="border",
                          align_corners=False).squeeze(0)
        x = (x - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.08, 0.08)
        return x.clamp_(0, 1)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            x = self._augment(x)
        if self.out_size and self.out_size != x.shape[-1]:
            x = F.interpolate(x.unsqueeze(0), size=self.out_size, mode="bilinear",
                              align_corners=False).squeeze(0)
        if self.rgb:
            x = x.expand(3, -1, -1)
            m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = (x - m) / s  # ImageNet statistics for pretrained backbones
        else:
            x = (x - self.mean) / self.std
        return x, self.y[i].float()


def make_loaders(size=28, augment=False, out_size=None, rgb=False, batch_size=128, seed=0):
    loaders = {}
    for split in ("train", "val", "test"):
        imgs, labels = load_split(split, size)
        ds = XrayDataset(imgs, labels, augment=(augment and split == "train"),
                         out_size=out_size, rgb=rgb)
        g = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    generator=g, num_workers=0)
    return loaders


# --------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """Basic model: 2 conv blocks + 1 hidden FC layer (~0.1M params at 28x28)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class DeepCNN(nn.Module):
    """Deeper, regularised CNN: 3 conv blocks (conv-BN-ReLU x2), dropout, global pooling."""

    def __init__(self, dropout=0.4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def resnet18(pretrained: bool, freeze_backbone: bool = False):
    """ResNet-18 with a 1-logit head. Input must be 3-channel ImageNet-normalised."""
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 1)
    if freeze_backbone:
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith("fc.")
    return _Squeeze(m)


class _Squeeze(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(1)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# -------------------------------------------------------------- metrics
@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys, loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logit = model(x)
        loss += F.binary_cross_entropy_with_logits(logit, y, reduction="sum").item()
        probs.append(torch.sigmoid(logit).cpu())
        ys.append(y.cpu())
        n += len(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int), loss / n


def compute_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                specificity=tn / (tn + fp), roc_auc=roc_auc_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=thr)


def best_f1_threshold(y, p):
    """Pick the threshold on VALIDATION data only."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid]))])


# ------------------------------------------------------------- training
def train_model(model, loaders, epochs=30, lr=1e-3, weight_decay=0.0, pos_weight=None,
                patience=8, name="model", verbose=True):
    """Train with Adam; select the epoch with the best validation ROC-AUC (early stopping).
    Test data is never touched here."""
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    pw = torch.tensor(pos_weight, device=DEVICE) if pos_weight else None
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_auc")}
    best_auc, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += ((logit > 0).float() == y).sum().item()
            n += len(y)
        pv, yv, vl = predict(model, loaders["val"])
        va = float(((pv >= 0.5) == yv).mean())
        auc = roc_auc_score(yv, pv)
        for k, v in zip(hist, (tl / n, tc / n, vl, va, auc)):
            hist[k].append(float(v))
        if verbose:
            print(f"[{name}] ep{ep:02d} train_loss={tl/n:.4f} acc={tc/n:.3f} | "
                  f"val_loss={vl:.4f} acc={va:.3f} auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = ep
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["train_seconds"] = time.time() - t0
    return model, hist

