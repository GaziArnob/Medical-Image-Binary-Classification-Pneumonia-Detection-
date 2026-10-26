"""Web demo for the pneumonia classifier (standard library HTTP server + the project's own dependencies).

    python pneumonia/app.py            # then open http://localhost:8000

Features: browse/filter test-set X-rays, Grad-CAM heatmaps, live decision-threshold slider with ROC marker,
side-by-side model comparison, results dashboard, batch upload with CSV export, and an optional local LLM
comment (Ollama). Research/education demo only: NOT for medical use.
"""
import base64
import csv
import io
import json
import os
import random
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from common import DEVICE, ROOT, DeepCNN, SmallCNN, load_split, resnet18

PORT = int(os.environ.get("PORT", 8000))
LABELS = {0: "Normal", 1: "Pneumonia"}
lock = threading.Lock()
FIG_DIR = ROOT / "results" / "figures"

# ------------------------------------------------------------------ models
# id -> (label, constructor, input kind, Grad-CAM target layer)
REGISTRY = {
    "deep_cnn_aug": ("Deep CNN + augmentation", DeepCNN, "gray28", lambda m: m.features[2][5]),
    "deep_cnn": ("Deep CNN (no augmentation)", DeepCNN, "gray28", lambda m: m.features[2][5]),
    "basic_cnn": ("Basic CNN", SmallCNN, "gray28", lambda m: m.features[4]),
    "resnet18_finetune_64": ("ResNet-18 fine-tuned (64 px)", lambda: resnet18(False), "rgb64", lambda m: m.m.layer4),
}
MODELS = {}
for mid, (label, ctor, kind, layer) in REGISTRY.items():
    f = ROOT / "models" / f"{mid}.pt"
    if f.exists():
        net = ctor().to(DEVICE)
        net.load_state_dict(torch.load(f, map_location=DEVICE))
        net.eval()
        MODELS[mid] = dict(label=label, net=net, kind=kind, layer=layer)
if not MODELS:
    sys.exit("No models found. Create them with:  python pneumonia/run_experiments.py --stage export_model")
DEFAULT = "deep_cnn_aug" if "deep_cnn_aug" in MODELS else next(iter(MODELS))

TEST_X, TEST_Y = load_split("test")  # 28x28 uint8: what the models see (same images as the reported test results)
try:
    TEST_X224, y224 = load_split("test", 224)  # same images at 224x224, only for a clearer display
    assert (y224 == TEST_Y).all()
except Exception:
    TEST_X224 = None


def prep(imgs28: np.ndarray, kind: str) -> torch.Tensor:
    """uint8 [N,28,28] -> model input, identical to the training-time preprocessing."""
    x = torch.from_numpy(imgs28).float().div(255.0).unsqueeze(1)
    if kind == "rgb64":
        x = F.interpolate(x, size=64, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((x - m) / s).to(DEVICE)
    return ((x - 0.5) / 0.5).to(DEVICE)


@torch.no_grad()
def probs_for(mid: str, imgs28: np.ndarray) -> np.ndarray:
    out = []
    for s in range(0, len(imgs28), 128):
        with lock:
            out.append(torch.sigmoid(MODELS[mid]["net"](prep(imgs28[s:s + 128], MODELS[mid]["kind"]))).cpu().numpy())
    return np.concatenate(out)


def predict_prob(mid: str, img28: np.ndarray) -> float:
    return float(probs_for(mid, img28[None])[0])


def metrics_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=div(tp + tn, len(y)), precision=prec, recall=rec,
                specificity=div(tn, tn + fp), f1=div(2 * prec * rec, prec + rec), threshold=t)


TEST_P = {mid: probs_for(mid, TEST_X) for mid in MODELS}  # computed once at start-up (624 images per model)
AUC = {mid: float(roc_auc_score(TEST_Y, p)) for mid, p in TEST_P.items()}


def get_model(q) -> str:
    mid = q.get("model", [DEFAULT])[0]
    return mid if mid in MODELS else DEFAULT


# ------------------------------------------------------------------ Grad-CAM
def gradcam_png(mid: str, img28: np.ndarray, display: np.ndarray) -> bytes:
    """Heatmap of the regions that push the prediction towards 'pneumonia', overlaid on the display image."""
    m = MODELS[mid]
    store = {}
    h = m["layer"](m["net"]).register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    try:
        with lock:
            logit = m["net"](prep(img28[None], m["kind"]))
            a = store["a"]
            g = torch.autograd.grad(logit.sum(), a)[0]
    finally:
        h.remove()
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1))[0].detach().cpu().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    size = display.shape[0]
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    gray = np.stack([display / 255.0] * 3, -1)
    out = (0.55 * gray + 0.45 * heat * (cam[..., None] > 0.05) + 0.45 * gray * (cam[..., None] <= 0.05))
    return to_png((np.clip(out, 0, 1) * 255).astype(np.uint8))


def to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def png_bytes(arr: np.ndarray, scale: int = 8) -> bytes:
    im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def display_image(i: int) -> np.ndarray:
    return TEST_X224[i] if TEST_X224 is not None else np.array(Image.fromarray(TEST_X[i]).resize((224, 224), Image.BICUBIC))


def read_upload(data: bytes):
    """Uploaded file -> (28x28 uint8 for the model, 224x224 uint8 for display)."""
    im = Image.open(io.BytesIO(data)).convert("L")
    return (np.array(im.resize((28, 28), Image.BILINEAR), dtype=np.uint8),
            np.array(im.resize((224, 224), Image.BICUBIC), dtype=np.uint8))


# ------------------------------------------------------------------ results dashboard
def results_summary() -> list:
    f = ROOT / "results" / "all_runs.csv"
    if not f.exists():
        return []
    groups = {}
    for r in csv.DictReader(open(f, newline="")):
        groups.setdefault(r["name"], []).append(r)
    out = []
    for name, rs in groups.items():
        row = dict(name=name, seeds=len(rs))
        for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
            v = np.array([float(r[f"test_thr05__{k}"]) for r in rs])
            row[k], row[k + "_sd"] = float(v.mean()), float(v.std())
        out.append(row)
    return sorted(out, key=lambda r: -r["roc_auc"])


# ------------------------------------------------------------------ optional free, local LLM comment (Ollama)
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VISION_HINTS = ("llava", "moondream", "-vl", "vl:", "vision", "minicpm-v", "gemma3", "bakllava", "granite3.2-vision")
VISION_PROMPT = (
    "This is a research/education demo using the public PneumoniaMNIST paediatric chest X-ray benchmark. "
    "Look at this chest X-ray and give an independent read. Reply with ONLY a JSON object: "
    '{"impression": "pneumonia" or "normal" or "uncertain", "confidence": "low" or "medium" or "high", '
    '"findings": "2-3 short sentences describing what you see"}. Be honest about uncertainty and image quality.')


def _ollama(path, payload=None, timeout=180):
    req = urllib.request.Request(OLLAMA + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def pick_model():
    """(name, is_vision). Vision models only if VERIFY_MODEL names one: the two free vision models I tested
    (moondream, qwen2.5vl:3b) were at chance level on PneumoniaMNIST, so the default is a text-only model."""
    if os.environ.get("VERIFY_MODEL"):
        m = os.environ["VERIFY_MODEL"]
        return m, any(h in m.lower() for h in VISION_HINTS)
    names = [m["name"] for m in _ollama("/api/tags", timeout=5)["models"]]
    text = [n for n in names if not any(h in n.lower() for h in VISION_HINTS)]
    if not text:
        raise RuntimeError("no text model installed")
    return text[0], False


def llm_second_opinion(png: bytes, mid: str, prob: float) -> dict:
    try:
        model_name, vision = pick_model()
    except Exception:
        return dict(error="Local LLM not reachable. Install Ollama (free) from ollama.com, start it, then run: "
                          "ollama pull qwen2.5:3b")
    mt = metrics_at(TEST_Y, TEST_P[mid], 0.5)
    try:
        if vision:
            msg = dict(role="user", content=VISION_PROMPT, images=[base64.b64encode(png).decode()])
        else:
            msg = dict(role="user", content=(
                f"A small CNN ({MODELS[mid]['label']}) estimates a {100 * prob:.1f}% probability of pneumonia on a chest X-ray "
                f"(threshold 0.5). On held-out data this CNN has {100 * mt['accuracy']:.0f}% accuracy, {100 * mt['recall']:.1f}% recall "
                f"and {100 * mt['specificity']:.0f}% specificity. "
                'Reply with ONLY a JSON object: {"findings": "2-3 short sentences on how much to trust this output and what a '
                'clinician would check next"}. You cannot see the image.'))
        out = _ollama("/api/chat", dict(model=model_name, messages=[msg], stream=False, format="json",
                                        options=dict(temperature=0)))["message"]["content"]
        res = json.loads(out)
        if vision and (len(str(res.get("findings", ""))) < 25 or "short sentences" in str(res.get("findings", "")).lower()):
            return dict(error=f"{model_name} did not produce a usable description of the image (it echoed the prompt).")
        res["impression"] = str(res.get("impression", "")).lower() if vision else None
        res["mode"] = "vision" if vision else "text"
        res["model"] = model_name
        return res
    except Exception as e:
        return dict(error=f"Local LLM request failed ({type(e).__name__}). Is Ollama running and the model installed?")


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        parts = u.path.strip("/").split("/")
        try:
            return self._get(u, q, parts)
        except (ValueError, IndexError, KeyError):
            return self._json(dict(error="bad request"), 400)

    def _get(self, u, q, parts):
        mid = get_model(q)
        t = float(q.get("t", ["0.5"])[0])
        if u.path == "/":
            return self._send((Path(__file__).parent / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/models":
            return self._json(dict(default=DEFAULT, n_test=len(TEST_Y), models=[
                dict(id=k, label=v["label"], auc=AUC[k], **metrics_at(TEST_Y, TEST_P[k], 0.5)) for k, v in MODELS.items()]))
        if u.path == "/api/threshold":
            return self._json(dict(auc=AUC[mid], **metrics_at(TEST_Y, TEST_P[mid], t)))
        if u.path == "/api/roc":
            fpr, tpr, _ = roc_curve(TEST_Y, TEST_P[mid])
            keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
            return self._json(dict(fpr=fpr[keep].tolist(), tpr=tpr[keep].tolist(), auc=AUC[mid]))
        if u.path == "/api/results":
            return self._json(results_summary())
        if u.path == "/api/samples":  # filter + paging over the 624 test images
            flt = q.get("filter", ["all"])[0]
            p, y = TEST_P[mid], TEST_Y
            pred = (p >= t).astype(int)
            mask = {"all": np.ones_like(y, bool), "normal": y == 0, "pneumonia": y == 1, "wrong": pred != y,
                    "fp": (pred == 1) & (y == 0), "fn": (pred == 0) & (y == 1)}[flt]
            idx = np.where(mask)[0]
            n = int(q.get("n", ["12"])[0])
            if q.get("shuffle", ["0"])[0] == "1":
                idx = np.array(random.sample(list(idx), min(n, len(idx))), dtype=int)
                page_idx = idx
            else:
                page = int(q.get("page", ["0"])[0])
                page_idx = idx[page * n:(page + 1) * n]
            return self._json(dict(total=int(mask.sum()), items=[
                dict(i=int(i), prob=float(p[i]), truth=LABELS[int(y[i])], pred=LABELS[int(pred[i])]) for i in page_idx]))
        if parts[:2] == ["api", "test"] and parts[-1].endswith(".png"):
            i = int(parts[-1][:-4])
            if "small" in q or TEST_X224 is None:
                return self._send(png_bytes(TEST_X[i]), "image/png")
            return self._send(png_bytes(TEST_X224[i], scale=1), "image/png")
        if parts[:2] == ["api", "predict_test"]:
            i = int(parts[-1])
            pr = float(TEST_P[mid][i])
            return self._json(dict(prob=pr, pred=LABELS[int(pr >= t)], truth=LABELS[int(TEST_Y[i])],
                                   correct=int(pr >= t) == int(TEST_Y[i])))
        if parts[:2] == ["api", "compare_test"]:
            i = int(parts[-1])
            return self._json([dict(id=k, label=v["label"], prob=float(TEST_P[k][i]),
                                    pred=LABELS[int(TEST_P[k][i] >= t)], correct=int(TEST_P[k][i] >= t) == int(TEST_Y[i]))
                               for k, v in MODELS.items()])
        if parts[:2] == ["api", "cam_test"] and parts[-1].endswith(".png"):
            i = int(parts[-1][:-4])
            return self._send(gradcam_png(mid, TEST_X[i], display_image(i)), "image/png")
        if parts[:2] == ["api", "verify_test"]:
            i = int(parts[-1])
            img = display_image(i)
            return self._json(llm_second_opinion(to_png(img), mid, float(TEST_P[mid][i])))
        if parts[:1] == ["figures"] and len(parts) == 2 and parts[1].endswith(".png"):
            f = FIG_DIR / Path(parts[1]).name  # no path traversal: only the file name is used
            if f.is_file():
                return self._send(f.read_bytes(), "image/png")
        return self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        mid = get_model(q)
        t = float(q.get("t", ["0.5"])[0])
        if u.path not in ("/api/predict", "/api/verify", "/api/cam"):
            return self._send(b"not found", "text/plain", 404)
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            img28, img224 = read_upload(data)
        except Exception:
            return self._json(dict(error="Could not read this file as an image."), 400)
        pr = predict_prob(mid, img28)
        if u.path == "/api/cam":
            return self._send(gradcam_png(mid, img28, img224), "image/png")
        if u.path == "/api/verify":
            return self._json(llm_second_opinion(to_png(img224), mid, pr))
        self._json(dict(prob=pr, pred=LABELS[int(pr >= t)], all=[
            dict(id=k, label=v["label"], prob=predict_prob(k, img28)) for k, v in MODELS.items()]))

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Demo running on http://localhost:{PORT}  (device: {DEVICE}, models: {', '.join(MODELS)}) - Ctrl+C to stop")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
