"""Run all PneumoniaMNIST experiments and write results/ (csv, figures).

Usage:
    python pneumonia/run_experiments.py --stage eda
    python pneumonia/run_experiments.py --stage basic        # Basic + Intermediate (28x28)
    python pneumonia/run_experiments.py --stage advanced     # pretrained vs scratch (28->64, 224)
    python pneumonia/run_experiments.py --stage report       # aggregate tables + figures
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

for _stream in (sys.stdout, sys.stderr):  # avoid UnicodeEncodeError on Windows consoles / pipes
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA_DIR, DEVICE, OUT_DIR, DeepCNN, SmallCNN, best_f1_threshold,
                    compute_metrics, count_params, load_split, make_loaders, predict,
                    resnet18, set_seed, train_model)

SEEDS = [0, 1, 2]
FIG = OUT_DIR / "figures"
INT_FIELDS = {"seed", "params", "trainable_params", "best_epoch", "tn", "fp", "fn", "tp"}
GROUPS = ("test_thr05", "test_valthr", "val_thr05")


def write_csv(path, rows):
    """Write a list of dicts (same keys) to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flatten_run(res):
    """One flat CSV row per run: metadata + '<group>__<metric>' columns."""
    row = {k: res[k] for k in ("name", "seed", "params", "trainable_params", "test_loss") if k in res}
    row.setdefault("seed", 0)
    if "history" in res:
        row["best_epoch"] = res["history"]["best_epoch"]
        row["train_seconds"] = round(res["history"]["train_seconds"], 1)
    for g in GROUPS:
        for k, v in res.get(g, {}).items():
            row[f"{g}__{k}"] = v
    return row


def save_run(res):
    stem = f"{res['name']}_s{res.get('seed', 0)}"
    write_csv(OUT_DIR / "runs" / f"{stem}.csv", [flatten_run(res)])
    if "history" in res:  # per-epoch training curves
        h = res["history"]
        cols = ("train_loss", "train_acc", "val_loss", "val_acc", "val_auc")
        write_csv(OUT_DIR / "history" / f"{stem}.csv",
                  [dict(epoch=i + 1, **{c: h[c][i] for c in cols}) for i in range(len(h["train_loss"]))])


# ------------------------------------------------------------------ EDA
def eda():
    FIG.mkdir(parents=True, exist_ok=True)
    stats = {}
    for s in ("train", "val", "test"):
        x, y = load_split(s)
        stats[s] = dict(n=len(y), normal=int((y == 0).sum()), pneumonia=int((y == 1).sum()),
                        pneumonia_frac=float(y.mean()), pixel_mean=float(x.mean()))
    write_csv(OUT_DIR / "dataset_stats.csv", [dict(split=k, **v) for k, v in stats.items()])
    print(stats)
    x, y = load_split("train")
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(2, 8, figsize=(12, 3.4))
    for r, c in enumerate((0, 1)):
        idx = rng.choice(np.where(y == c)[0], 8, replace=False)
        for j, i in enumerate(idx):
            ax[r, j].imshow(x[i], cmap="gray", vmin=0, vmax=255)
            ax[r, j].axis("off")
        ax[r, 0].set_title(["Normal", "Pneumonia"][c], loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "samples.png", dpi=150)
    plt.close()


# ------------------------------------------------------- one experiment
def run_one(name, build, seed, size=28, augment=False, out_size=None, rgb=False,
            epochs=30, lr=1e-3, wd=0.0, batch=128, pos_weight=None):
    set_seed(seed)
    loaders = make_loaders(size=size, augment=augment, out_size=out_size, rgb=rgb,
                           batch_size=batch, seed=seed)
    model = build()
    n_params, n_train = count_params(model), count_params(model, True)
    model, hist = train_model(model, loaders, epochs=epochs, lr=lr, weight_decay=wd,
                              pos_weight=pos_weight, name=f"{name}-s{seed}")
    pv, yv, _ = predict(model, loaders["val"])
    thr = best_f1_threshold(yv, pv)  # tuned on validation only
    pt, yt, test_loss = predict(model, loaders["test"])
    res = dict(name=name, seed=seed, params=n_params, trainable_params=n_train,
               history=hist, test_loss=test_loss,
               test_thr05=compute_metrics(yt, pt, 0.5),
               test_valthr=compute_metrics(yt, pt, thr),
               val_thr05=compute_metrics(yv, pv, 0.5))
    np.savez_compressed(OUT_DIR / "preds" / f"{name}_s{seed}.npz", p=pt, y=yt)
    save_run(res)
    m = res["test_thr05"]
    print(f"==> {name} seed{seed}: test acc={m['accuracy']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.4f}", flush=True)
    return res


def majority_baseline():
    _, ytr = load_split("train")
    _, yte = load_split("test")
    maj = int(np.bincount(ytr).argmax())
    p = np.full(len(yte), 0.9 if maj == 1 else 0.1)
    p = p + np.random.RandomState(0).uniform(-1e-3, 1e-3, len(p))  # AUC is 0.5 by construction
    m = compute_metrics(yte, p, 0.5)
    m["roc_auc"] = 0.5
    save_run(dict(name="majority", seed=0, test_thr05=m))
    print("majority:", m)


def stage_basic():
    majority_baseline()
    for s in SEEDS:
        run_one("basic_cnn", SmallCNN, s, epochs=40)
        run_one("deep_cnn", DeepCNN, s, epochs=40, wd=1e-4)
        run_one("deep_cnn_aug", DeepCNN, s, augment=True, epochs=60, wd=1e-4)
        # sensitivity: class-weighted loss (positive:negative is ~3:1 in train, so weight normals up)
        run_one("deep_cnn_aug_posw0.5", DeepCNN, s, augment=True, epochs=60, wd=1e-4, pos_weight=0.5)


def stage_advanced():
    for s in SEEDS:
        # 28x28 inputs upsampled to 64x64 and converted to 3 channels
        kw = dict(size=28, out_size=64, rgb=True, epochs=25, batch=128)
        run_one("resnet18_scratch_64", lambda: resnet18(False), s, lr=1e-3, **kw)
        run_one("resnet18_linprobe_64", lambda: resnet18(True, freeze_backbone=True), s, lr=1e-3, **kw)
        run_one("resnet18_finetune_64", lambda: resnet18(True), s, lr=3e-4, **kw)


def stage_advanced224():
    """Needs the full 224x224 download (pneumoniamnist_224.npz)."""
    for s in SEEDS[:2]:
        kw = dict(size=224, rgb=True, epochs=8, batch=64)
        run_one("resnet18_scratch_224", lambda: resnet18(False), s, lr=1e-3, **kw)
        run_one("resnet18_finetune_224", lambda: resnet18(True), s, lr=1e-4, **kw)


# models saved for the web demo (seed 0, same settings as the experiments); existing files are kept
EXPORT = {
    "basic_cnn": dict(build=SmallCNN, load=dict(), fit=dict(epochs=40)),
    "deep_cnn": dict(build=DeepCNN, load=dict(), fit=dict(epochs=40, weight_decay=1e-4)),
    "deep_cnn_aug": dict(build=DeepCNN, load=dict(augment=True), fit=dict(epochs=60, weight_decay=1e-4)),
    "resnet18_finetune_64": dict(build=lambda: resnet18(True), load=dict(out_size=64, rgb=True),
                                 fit=dict(epochs=25, lr=3e-4)),
}


def export_model():
    """Train (seed 0) and save the models used by the web demo to models/<name>.pt."""
    out = OUT_DIR.parent / "models"
    out.mkdir(exist_ok=True)
    for name, cfg in EXPORT.items():
        if (out / f"{name}.pt").exists():
            print("exists, skipping:", name)
            continue
        set_seed(0)
        loaders = make_loaders(size=28, batch_size=128, seed=0, **cfg["load"])
        model, _ = train_model(cfg["build"](), loaders, name=f"export-{name}", **cfg["fit"])
        pt, yt, _ = predict(model, loaders["test"])
        torch.save(model.state_dict(), out / f"{name}.pt")
        print(f"saved models/{name}.pt; test:", {k: round(float(v), 3) for k, v in compute_metrics(yt, pt).items() if isinstance(v, (float, np.floating))})


# --------------------------------------------------------------- report
def load_runs():
    """Rebuild {name: [run dict, ...]} from results/runs/*.csv and results/history/*.csv."""
    runs = {}
    for f in sorted((OUT_DIR / "runs").glob("*.csv")):
        row = read_csv(f)[0]
        r = {}
        for k, v in row.items():
            if "__" in k:
                g, m = k.split("__", 1)
                r.setdefault(g, {})[m] = int(float(v)) if m in INT_FIELDS else float(v)
            elif k == "name":
                r[k] = v
            else:
                r[k] = int(float(v)) if k in INT_FIELDS else float(v)
        h = OUT_DIR / "history" / f.name
        if h.exists():
            rows = read_csv(h)
            r["history"] = {c: [float(x[c]) for x in rows] for c in rows[0] if c != "epoch"}
            r["history"]["best_epoch"] = r["best_epoch"]
            r["history"]["train_seconds"] = r["train_seconds"]
        runs.setdefault(r["name"], []).append(r)
    return runs


def stage_report():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    keys = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    lines = ["| Model | Params (trainable) | Thr | " + " | ".join(keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for name, rs in runs.items():
        if name == "majority":
            lines.append("| majority | 0 | 0.5 | " + " | ".join(f"{rs[0]['test_thr05'][k]:.3f}" for k in keys) + " |")
            continue
        for tag, field in (("0.5", "test_thr05"), ("val-tuned", "test_valthr")):
            cells = []
            for k in keys:
                v = np.array([r[field][k] for r in rs])
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v[0]:.3f}")
            lines.append(f"| {name} | {rs[0]['params']:,} ({rs[0]['trainable_params']:,}) | {tag} | "
                         + " | ".join(cells) + " |")
    write_csv(OUT_DIR / "all_runs.csv", [flatten_run(r) for rs in runs.values() for r in rs])
    (OUT_DIR / "results_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # training curves (seed 0) for the CNN models
    for name in [n for n in runs if n in ("basic_cnn", "deep_cnn", "deep_cnn_aug")]:
        h = runs[name][0]["history"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(h["train_loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(h["train_acc"], label="train"); ax[1].plot(h["val_acc"], label="val")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        ax[0].axvline(h["best_epoch"] - 1, ls=":", c="gray")
        plt.suptitle(name); plt.tight_layout()
        plt.savefig(FIG / f"curves_{name}.png", dpi=150); plt.close()

    # ROC curves + confusion matrices (seed 0)
    plt.figure(figsize=(5, 4.5))
    for name in runs:
        f = OUT_DIR / "preds" / f"{name}_s0.npz"
        if not f.exists():
            continue
        d = np.load(f)
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=6); plt.title("Test ROC (seed 0)"); plt.tight_layout()
    plt.savefig(FIG / "roc.png", dpi=150); plt.close()

    for name in ("basic_cnn", "deep_cnn_aug", "resnet18_finetune_64"):
        if name not in runs:
            continue
        m = runs[name][0]["test_thr05"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        plt.figure(figsize=(3.4, 3.2))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["Normal", "Pneum."]); plt.yticks([0, 1], ["Normal", "Pneum."])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(name, fontsize=9); plt.tight_layout()
        plt.savefig(FIG / f"cm_{name}.png", dpi=150); plt.close()

    # error examples for the best-looking small model
    if "deep_cnn_aug" in runs:
        d = np.load(OUT_DIR / "preds" / "deep_cnn_aug_s0.npz")
        x, _ = load_split("test")
        pred = (d["p"] >= 0.5).astype(int)
        fn = np.where((d["y"] == 1) & (pred == 0))[0][:6]
        fp = np.where((d["y"] == 0) & (pred == 1))[0][:6]
        fig, ax = plt.subplots(2, 6, figsize=(11, 4))
        for r, (idx, t) in enumerate(((fn, "Missed pneumonia (FN)"), (fp, "False alarm (FP)"))):
            for j in range(6):
                ax[r, j].axis("off")
                if j < len(idx):
                    ax[r, j].imshow(x[idx[j]], cmap="gray")
                    ax[r, j].set_title(f"p={d['p'][idx[j]]:.2f}", fontsize=8)
            ax[r, 0].text(-0.1, 1.25, t, transform=ax[r, 0].transAxes, fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / "errors.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eda", "basic", "advanced", "advanced224", "export_model", "report"])
    a = ap.parse_args()
    for d in ("runs", "history", "preds", "figures"):
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    {"eda": eda, "basic": stage_basic, "advanced": stage_advanced, "advanced224": stage_advanced224, "export_model": export_model, "report": stage_report}[a.stage]()
