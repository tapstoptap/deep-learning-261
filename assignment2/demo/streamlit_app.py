"""Food-101 demo (Streamlit): frozen ResNet-50 (E2) vs fully fine-tuned ResNet-50 (E3).

Group choco (99-PTQ) — CO3133 Assignment 02.
Upload a food photo; both models predict one of 101 Food-101 classes.

Checkpoints resolve in this order:
  1. Local file under demo/checkpoints/ (local development).
  2. Hugging Face model repo `HF_CKPT_REPO` (Streamlit Cloud deployment;
     e.g. Themenli/food101-demo-checkpoints). Model repos on HF Hub are free
     even though free CPU Spaces are no longer available to new accounts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

BASE_DIR = Path(__file__).resolve().parent
CLASSES: list[str] = json.loads((BASE_DIR / "food101_classes.json").read_text(encoding="utf-8"))
NUM_CLASSES = len(CLASSES)

HF_CKPT_REPO = os.environ.get("HF_CKPT_REPO", "Themenli/food101-demo-checkpoints")
# The checkpoints currently live in a public HF *Space* (repo_type="space");
# if you later move them to a proper HF *model* repo, set HF_CKPT_REPO_TYPE=model.
HF_CKPT_REPO_TYPE = os.environ.get("HF_CKPT_REPO_TYPE", "space")

MODEL_INFO = {
    "e3": {
        "file": "e3_resnet50_best.pt",
        "title": "ResNet-50 full fine-tune (E3)",
        "metrics": "test Macro-F1 86.10%, accuracy 86.17%",
    },
    "e2": {
        "file": "e2_resnet50_frozen_best.pt",
        "title": "ResNet-50 frozen backbone (E2)",
        "metrics": "test Macro-F1 66.36%, accuracy 66.70%",
    },
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResNet50Classifier(nn.Module):
    """Same architecture as src/models/pretrained.py in the course repo."""

    def __init__(self, num_classes: int = 101, dropout: float = 0.2) -> None:
        super().__init__()
        # weights=None: the trained checkpoint is loaded right after, so the
        # ImageNet download would be wasted on a free 1 GB container.
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def resolve_ckpt(filename: str) -> str | None:
    local = BASE_DIR / "checkpoints" / filename
    if local.exists():
        return str(local)
    try:
        return hf_hub_download(HF_CKPT_REPO, filename, repo_type=HF_CKPT_REPO_TYPE)
    except Exception:
        return None


@st.cache_resource
def load_model(key: str):
    ckpt_path = resolve_ckpt(MODEL_INFO[key]["file"])
    if ckpt_path is None:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet50Classifier(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(model, image: Image.Image) -> list[tuple[str, float]]:
    with torch.no_grad():
        x = EVAL_TRANSFORM(image).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1)[0]
    ranked = sorted(
        ((CLASSES[i], float(probs[i])) for i in range(NUM_CLASSES)),
        key=lambda kv: -kv[1],
    )
    return ranked[:6]


def main() -> None:
    st.set_page_config(
        page_title="Food-101 · Group choco (CO3133 A2)",
        page_icon="🍕",
        layout="wide",
    )
    st.title("Food-101 Classification · Group choco (CO3133 Assignment 02)")
    st.write(
        "Upload a food photo. Both ResNet-50 models predict one of 101 Food-101 "
        "classes. Everything runs on CPU."
    )

    example_dir = BASE_DIR / "examples"
    examples = sorted(example_dir.glob("*.jpg")) if example_dir.exists() else []

    left, right = st.columns([1, 2])
    with left:
        uploaded = st.file_uploader("Your photo", type=["jpg", "jpeg", "png", "webp"])
        if examples:
            pick = st.radio(
                "Examples",
                options=["(upload your own)"] + [f.stem for f in examples],
                horizontal=False,
            )
        else:
            pick = "(upload your own)"

    image: Image.Image | None = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif pick != "(upload your own)":
        image = Image.open(example_dir / f"{pick}.jpg").convert("RGB")

    with right:
        if image is None:
            st.info("Upload a photo or pick an example on the left.")
            return
        st.image(image, width=360)

        cols = st.columns(2)
        for col, key in zip(cols, ("e3", "e2")):
            info = MODEL_INFO[key]
            with col:
                st.subheader(info["title"])
                st.caption(info["metrics"])
                with st.spinner("Loading model…"):
                    model = load_model(key)
                if model is None:
                    st.info(
                        f"Checkpoint `{info['file']}` not available "
                        f"(neither local nor in `{HF_CKPT_REPO}`)."
                    )
                    continue
                top6 = predict(model, image)
                st.markdown(f"## {top6[0][0].replace('_', ' ')}")
                for label, p in top6:
                    st.progress(min(max(p, 0.0), 1.0), text=f"{label}  {p:.1%}")

    st.divider()
    st.caption(
        "Trained on the official Food-101 split (68,175 train / 7,575 validation / "
        "25,250 test). Code, configs, and checkpoint contract: "
        "github.com/tapstoptap/deep-learning-261 · assignment2/README.md · CHECKPOINTS.md."
    )


if __name__ == "__main__":
    main()
