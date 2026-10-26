# Medical Image Binary Classification — PneumoniaMNIST (Project 4)

Normal vs. pneumonia chest X-rays (positive class = pneumonia) with CNNs and transfer learning.
Levels attempted: **Basic, Intermediate, Advanced**. For research/education only, not clinical use.

## Quick start (any PC: Windows, macOS, Linux)
Needs Python 3.11 or newer and an internet connection for the first run. A GPU is optional (everything also runs on CPU, only slower).
```
git clone https://github.com/GaziArnob/Medical-Image-Binary-Classification-Pneumonia-Detection-.git
cd Medical-Image-Binary-Classification-Pneumonia-Detection-
python -m venv .venv
.venv\Scriptsctivate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # NVIDIA GPU: see the comment at the top of requirements.txt first
python pneumonia/app.py           # web demo -> open http://localhost:8000
```
The demo works straight away because the trained models are in `models/`; the 28x28 test data (~4 MB) downloads automatically.
Images are shown enlarged from 28x28. For sharper display (optional, 214 MB download) run once:
`python -c "from medmnist import PneumoniaMNIST; PneumoniaMNIST(split='test', download=True, size=224, root='data')"`

## Setup
```
pip install -r requirements.txt
```
Data: downloaded automatically by `medmnist` (`download=True`) into `data/`
(28x28 file ~4 MB; 224x224 file is a large download from Zenodo). Supplied train/val/test splits are kept.
License: CC BY 4.0. Cite: Yang et al., *MedMNIST v2*, Scientific Data 2023; Kermany et al., Cell 2018.

## Run
```
python pneumonia/run_experiments.py --stage eda        # class counts, sample grid
python pneumonia/run_experiments.py --stage basic      # majority baseline, Basic CNN, deep CNN, augmentation, class weight
python pneumonia/run_experiments.py --stage advanced     # ResNet-18 scratch vs pretrained (28px upsampled to 64px)
python pneumonia/run_experiments.py --stage advanced224  # same at native 224px (needs the 224 download)
python pneumonia/run_experiments.py --stage report     # tables + figures in results/
```
Seeds 0,1,2 (224px: seeds 0,1). Early stopping / model selection uses validation ROC-AUC; the
decision threshold is either 0.5 or tuned on validation; the test split is used only for final evaluation.

## Web demo (optional)
```
python pneumonia/run_experiments.py --stage export_model   # trains + saves 4 demo models to models/ (seed 0, ~5 min on GPU)
python pneumonia/app.py                                    # open http://localhost:8000
```
Features: browse the 624 test X-rays with filters (all / true class / mistakes / false alarms / missed pneumonia) and jump-to-index;
Grad-CAM heatmap (where the model looks); live decision-threshold slider with confusion matrix, metrics and ROC marker;
side-by-side comparison of 4 models (Basic CNN, Deep CNN, Deep CNN + augmentation, fine-tuned ResNet-18); results dashboard
(report tables + figures); upload one image or a batch (CSV export). Research/education only; not for medical use.
The saved demo models are single seed-0 copies, so their numbers can differ slightly from the multi-seed tables in the report.

Optional free LLM (local, no API key): install [Ollama](https://ollama.com) and a small text model (e.g. `ollama pull qwen2.5:3b`);
the "Ask the local LLM" button then gives a short comment on how far to trust the CNN output. This model cannot see the image.

Vision LLMs were tried as an independent "second reader" and did NOT work: on 24 balanced test images `moondream` and
`qwen2.5vl:3b` were at chance level (accuracy 0.50 and 0.54; they answered "no pneumonia" for almost every pneumonia image).
The app therefore uses vision models only when you set `VERIFY_MODEL=<model>` explicitly. Small LLMs are not validated for radiology.

## Layout
- `pneumonia/common.py` — data loading, augmentation, models, training loop, metrics
- `pneumonia/run_experiments.py` — experiment stages and report generation
- `results/` — `all_runs.csv` (every run), `runs/` + `history/` (per-run CSV), `preds/`, `results_table.md`, `figures/`
