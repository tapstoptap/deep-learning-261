"""Food-101 demo: frozen ResNet-50 (E2) vs fully fine-tuned ResNet-50 (E3).

Group choco (99-PTQ) — CO3133 Assignment 02.
Upload a food photo; both models predict one of 101 Food-101 classes.
Checkpoints are produced by src/train.py in the course repository:
  E3: configs/pretrained.yaml          (freeze_backbone: false)
  E2: configs/pretrained_frozen.yaml   (freeze_backbone: true)
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

BASE_DIR = Path(__file__).resolve().parent
CLASSES: list[str] = json.loads((BASE_DIR / "food101_classes.json").read_text(encoding="utf-8"))
NUM_CLASSES = len(CLASSES)

E2_CKPT = BASE_DIR / "checkpoints" / "e2_resnet50_frozen_best.pt"
E3_CKPT = BASE_DIR / "checkpoints" / "e3_resnet50_best.pt"

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
        # ImageNet download would be wasted (matters on a free CPU Space).
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(ckpt_path: Path) -> ResNet50Classifier:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet50Classifier(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


MODELS: dict[str, ResNet50Classifier] = {}


def predict(image, model_key: str):
    if model_key not in MODELS:
        return None
    if image is None:
        return {}
    model = MODELS[model_key]
    with torch.no_grad():
        x = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}


def predict_both(image):
    return predict(image, "e3"), predict(image, "e2")


def predict_e3_only(image):
    return predict(image, "e3")


def main() -> None:
    if E3_CKPT.exists():
        MODELS["e3"] = load_model(E3_CKPT)
    if E2_CKPT.exists():
        MODELS["e2"] = load_model(E2_CKPT)
    if not MODELS:
        raise FileNotFoundError(f"No checkpoints found in {BASE_DIR / 'checkpoints'}")

    example_files = sorted((BASE_DIR / "examples").glob("*.jpg"))

    with gr.Blocks(title="Food-101 · Group choco (CO3133 A2)") as demo:
        gr.Markdown(
            "# Food-101 Classification · Group choco (CO3133 Assignment 02)\n"
            "Upload a food photo. Both ResNet-50 models predict one of 101 Food-101 classes. "
            "Everything runs on CPU in this Space."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Your photo")
                btn = gr.Button("Predict", variant="primary")
                if example_files:
                    gr.Examples(examples=[[str(f)] for f in example_files], inputs=inp)
        with gr.Row():
            e3_out = gr.Label(
                num_top_classes=6,
                label="ResNet-50 full fine-tune (E3) · test Macro-F1 86.10%, accuracy 86.17%",
            )
            if "e2" in MODELS:
                e2_out = gr.Label(
                    num_top_classes=6,
                    label="ResNet-50 frozen backbone (E2) · test Macro-F1 66.36%, accuracy 66.70%",
                )
                btn.click(fn=predict_both, inputs=inp, outputs=[e3_out, e2_out])
            else:
                gr.Markdown(
                    "*ResNet-50 frozen backbone (E2): checkpoint not bundled in this deployment.*"
                )
                btn.click(fn=predict_e3_only, inputs=inp, outputs=[e3_out])
        gr.Markdown(
            "Trained on the official Food-101 split (68,175 train / 7,575 validation / 25,250 test). "
            "Code, configs, and checkpoint contract: "
            "[deep-learning-261](https://github.com/tapstoptap/deep-learning-261) · "
            "`assignment2/README.md` · `CHECKPOINTS.md`."
        )

    demo.launch()


if __name__ == "__main__":
    main()
