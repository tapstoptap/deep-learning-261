# Food-101 demo

**Live demo:** <https://deep-learning-261-ahricdx9pw4yfwaqmrmx7e.streamlit.app/>

Web demo for Assignment 2: upload a food photo and compare the two ResNet-50
checkpoints side by side — E3 (full fine-tune, test Macro-F1 86.10% /
accuracy 86.17%) and E2 (frozen backbone, test Macro-F1 66.36% / accuracy
66.70%). Both predict one of the 101 Food-101 classes; the app shows the top-6
probabilities for each model.

Two interchangeable frontends are included:

- `streamlit_app.py` — Streamlit app (default, free to host on Streamlit Cloud).
- `app.py` — Gradio app (same models, for local use or a paid HF Space).

Checkpoints are ~95 MB each and are **not** committed to the course GitHub
repo. The app loads them from a local `checkpoints/` folder if present, and
otherwise downloads them from a public Hugging Face repo
(`Themenli/food101-demo-checkpoints`). Override with the `HF_CKPT_REPO`
environment variable; if the checkpoints live in a proper HF **model** repo
instead of a Space, also set `HF_CKPT_REPO_TYPE=model` (default is `space`).

## Files

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Streamlit app (model definition, transforms, HF checkpoint download) |
| `app.py` | Gradio app (self-contained alternative) |
| `requirements.txt` | CPU-only torch/torchvision + Streamlit + huggingface_hub |
| `food101_classes.json` | 101 class names in Food-101 `classes.txt` order |
| `examples/*.jpg` | Sample Food-101 validation images used as one-click examples |
| `checkpoints/e3_resnet50_best.pt` | E3 best checkpoint (copy of `assignment2/checkpoints/resnet50_best.pt`) |
| `checkpoints/e2_resnet50_frozen_best.pt` | E2 best checkpoint (from Kaggle Outputs: `checkpoints/resnet50_frozen/resnet50_best.pt`) |
| `.gitattributes` | Routes `*.pt` through Git LFS (required by HF repos) |

If the E2 checkpoint is missing, the app automatically shows only E3 with a
note instead of the second panel.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py   # http://127.0.0.1:8501
# or the Gradio version (needs gradio installed separately):
pip install "gradio>=5.0"
python app.py                    # http://127.0.0.1:7860
```

## Deploy (free): Streamlit Cloud + HF checkpoint host

Hugging Face Spaces now requires a paid plan for the Gradio SDK on new free
accounts, so the recommended free path is **Streamlit Cloud** for the app plus
a free **HF repo** for the two checkpoints.

1. **Host the checkpoints** (one time): the two `.pt` files live in the public
   HF repo `Themenli/food101-demo-checkpoints`. To use your own, upload both
   `.pt` files to any public HF **Space** (web UI "Files" → "Add file") or a
   proper HF **Model** repo, and note the repo id.
2. **Push the demo to GitHub**: commit `assignment2/demo/` (without
   `checkpoints/`) to the course repo branch.
3. **Create the app**: share.streamlit.io → New app → pick the GitHub repo and
   branch → main file path `assignment2/demo/streamlit_app.py` → Deploy.
   Streamlit installs `requirements.txt` automatically.
4. **Point the app at your checkpoint repo** (optional): in the app's
   Settings → Secrets, add `HF_CKPT_REPO = <your-user>/<repo>`. If it is a
   **Model** repo (not a Space), also add `HF_CKPT_REPO_TYPE = model`.
   Without these, the app uses the default Space `Themenli/food101-demo-checkpoints`.
5. First prediction is the slowest (downloads + loads both checkpoints, then
   CPU warm-up).

## Alternative: Hugging Face Spaces (Gradio)

Only available on a paid HF plan for new accounts. Create a Gradio Space,
clone it, and copy `app.py`, `requirements.txt`, `food101_classes.json`,
`.gitattributes`, `examples/`, and `checkpoints/` in; push and the Space
builds automatically.

## Provenance

Checkpoints are produced by `src/train.py` in the course repository
(`configs/pretrained.yaml` for E3, `configs/pretrained_frozen.yaml` for E2).
Checkpoint binaries are deliberately not committed to the course GitHub repo;
see `assignment2/CHECKPOINTS.md` for the reconstruction contract.
