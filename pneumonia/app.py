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
import re
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

for _stream in (sys.stdout, sys.stderr):  # avoid UnicodeEncodeError on Windows consoles / pipes
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
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
# Optional sharper display images. Used only if the 224x224 file is already on disk (it is a 214 MB download, so the demo
# never downloads it by itself); otherwise the 28x28 images are shown enlarged with smooth interpolation.
TEST_X224 = None
if (ROOT / "data" / "pneumoniamnist_224.npz").exists():
    try:
        TEST_X224, _y224 = load_split("test", 224)
        assert (_y224 == TEST_Y).all()
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


# ------------------------------------------------------------------ optional LLM comment
# Default backend works on any PC with only `pip install`: a small open model (Hugging Face transformers), downloaded once on
# first use (~1 GB, from huggingface.co; behind a firewall set HF_ENDPOINT=https://hf-mirror.com), run on CPU, no API key.
# If it cannot be loaded (no internet, not installed) a built-in template text is shown instead, clearly labelled.
# Setting VERIFY_MODEL=<ollama model> switches to a local Ollama server (also allows vision models).
HF_LABEL = "Qwen/Qwen2.5-0.5B-Instruct"
_LOCAL_LLM = ROOT / "models" / "llm" / "qwen2.5-0.5b-instruct"  # copy this folder to another PC to use the LLM without internet
HF_MODEL = os.environ.get("HF_MODEL", str(_LOCAL_LLM) if (_LOCAL_LLM / "config.json").exists() else HF_LABEL)
_hf = {}
llm_lock = threading.Lock()
LLM_SYSTEM = ("You write short, careful comments for a student research demo of a chest X-ray classifier. "
              "You are not a doctor: never give a diagnosis or treatment advice.")


def _hf_load():
    if "model" not in _hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # imported lazily: the rest of the app needs no transformers
        _hf["tok"] = AutoTokenizer.from_pretrained(HF_MODEL)
        _hf["model"] = AutoModelForCausalLM.from_pretrained(HF_MODEL).float().eval()
    return _hf["tok"], _hf["model"]


def _cnn_summary(mid: str, prob: float, mt: dict) -> str:
    return (f"A small CNN ({MODELS[mid]['label']}) estimates a {100 * prob:.1f}% probability of pneumonia on one chest X-ray "
            f"(decision threshold 0.5). On the {len(TEST_Y)} held-out test images this CNN has {100 * mt['accuracy']:.0f}% accuracy, "
            f"{100 * mt['recall']:.1f}% recall and {100 * mt['specificity']:.0f}% specificity.")


def hf_comment(facts: str) -> str:
    """Let the LLM reword the fixed facts in plain English. It may not add numbers (checked); its wording can still be
    wrong, so the app always shows the facts next to it."""
    tok, model = _hf_load()
    msgs = [{"role": "system", "content": LLM_SYSTEM + " Keep every number exactly as written and add no new numbers or advice."},
            {"role": "user", "content": "Rewrite in at most 3 short sentences, plain text, no lists:\n" + facts}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with llm_lock, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=90, do_sample=False)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    nums = lambda t: set(re.findall(r"\d+(?:\.\d+)?", t))
    if nums(text) - nums(facts):
        raise ValueError("the LLM added numbers that are not in the data")
    return text


def template_comment(prob: float, mt: dict) -> str:
    """Fallback when no LLM can be loaded: fixed text built from the real test metrics (not generated by a model)."""
    if prob >= 0.5:
        return (f"The CNN predicts pneumonia ({100 * prob:.1f}%). On the test images {100 * mt['precision']:.0f}% of the images it flagged "
                f"were truly pneumonia and about {100 * (1 - mt['specificity']):.0f}% of normal images were flagged too, so a positive "
                "result is not conclusive: a clinician should read the image and use other clinical information.")
    return (f"The CNN predicts normal ({100 * prob:.1f}% pneumonia probability). It still misses about {100 * (1 - mt['recall']):.1f}% "
            "of pneumonia cases on the test images, so a negative result does not rule pneumonia out and a clinician should "
            "confirm it.")


# --- optional Ollama backend (only when VERIFY_MODEL is set)
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


def ollama_opinion(png: bytes, mid: str, prob: float, mt: dict) -> dict:
    name = os.environ["VERIFY_MODEL"]
    vision = any(h in name.lower() for h in VISION_HINTS)
    try:
        if vision:  # note: the two free vision models I tested (moondream, qwen2.5vl:3b) were at chance level here
            msg = dict(role="user", content=VISION_PROMPT, images=[base64.b64encode(png).decode()])
        else:
            msg = dict(role="user", content=_cnn_summary(mid, prob, mt) + ' Reply with ONLY a JSON object: {"findings": "2-3 short '
                       'sentences on how much to trust this output and what a clinician would check next"}. You cannot see the image.')
        out = _ollama("/api/chat", dict(model=name, messages=[msg], stream=False, format="json", options=dict(temperature=0)))["message"]["content"]
        res = json.loads(out)
        if vision and (len(str(res.get("findings", ""))) < 25 or "short sentences" in str(res.get("findings", "")).lower()):
            return dict(error=f"{name} did not produce a usable description of the image (it echoed the prompt).")
        res["impression"] = str(res.get("impression", "")).lower() if vision else None
        res["mode"] = "vision" if vision else "text"
        res["model"] = name
        res["facts"] = template_comment(prob, mt)
        return res
    except Exception as e:
        return dict(error=f"Ollama request failed ({type(e).__name__}). Is Ollama running and the model '{name}' installed?")


def llm_second_opinion(png: bytes, mid: str, prob: float) -> dict:
    mt = metrics_at(TEST_Y, TEST_P[mid], 0.5)
    if os.environ.get("VERIFY_MODEL"):
        return ollama_opinion(png, mid, prob, mt)
    facts = template_comment(prob, mt)  # computed from the real test metrics: this is the reliable part
    try:
        return dict(facts=facts, findings=hf_comment(facts), impression=None, mode="text", model=HF_LABEL if HF_MODEL == str(_LOCAL_LLM) else HF_MODEL)
    except ValueError as e:  # the LLM answered but broke the "no new numbers" rule
        return dict(facts=facts, findings=None, impression=None, mode="template", model=HF_LABEL,
                    note="The LLM's wording was rejected because it added numbers that are not in the data. Only the facts are shown.")
    except Exception as e:
        return dict(facts=facts, findings=None, impression=None, mode="template", model="built-in template",
                    note=f"The LLM wording is not available ({type(e).__name__}: {str(e)[:160]}). Needs: pip install transformers, and either one "
                         f"download of {HF_LABEL} from huggingface.co (mirror: HF_ENDPOINT) or the folder models/llm/qwen2.5-0.5b-instruct copied from another PC. "
                         "Only the facts are shown.")


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
            if "small" in q:  # what the models actually see (28x28, enlarged without smoothing)
                return self._send(png_bytes(TEST_X[i]), "image/png")
            return self._send(to_png(display_image(i)), "image/png")
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
