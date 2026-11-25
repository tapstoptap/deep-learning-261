"""Cache frozen encoder features: ResNet-50 pooled (per image) and BERT [CLS] (per question).

Reads images and question text only; no answers are loaded.
Usage: python src/extract_features.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel, AutoTokenizer

from datasets.vqa import DATA, FEATURES, SOURCE, coco_name, split_ids

DEV = "cuda"
TF = ResNet50_Weights.IMAGENET1K_V2.transforms(resize_size=256, crop_size=224)


class Images(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, image_id = self.items[i]
        path = DATA / "images" / src / coco_name(src, image_id)
        return TF(Image.open(path).convert("RGB"))


def main():
    FEATURES.mkdir(exist_ok=True)
    ids = {s: split_ids(s) for s in SOURCE}

    items = [(SOURCE[s], int(i)) for s in SOURCE for i in sorted(ids[s].image_id.unique())]
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(DEV)
    dl = torch.utils.data.DataLoader(Images(items), batch_size=64, num_workers=8)
    t0, out = time.time(), []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in dl:
            out.append(net(batch.to(DEV)).half().cpu().numpy())
    img = np.concatenate(out)
    np.save(FEATURES / "image.npy", img)
    pd.DataFrame(items, columns=["source", "image_id"]).to_csv(FEATURES / "image_index.csv", index=False)
    print(f"image features {img.shape} in {time.time() - t0:.0f} s")
    del net

    text = {}
    for src in sorted(set(SOURCE.values())):
        qj = json.loads((DATA / f"v2_OpenEnded_mscoco_{src}_questions.json").read_text())
        text.update({q["question_id"]: q["question"] for q in qj["questions"]})
    qids = [int(q) for s in SOURCE for q in ids[s].question_id]
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").eval().to(DEV)
    t0, out = time.time(), []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(qids), 256):
            enc = tok([text[q] for q in qids[i:i + 256]], padding="max_length", truncation=True,
                      max_length=32, return_tensors="pt").to(DEV)
            out.append(bert(**enc).last_hidden_state[:, 0].half().cpu().numpy())
    txt = np.concatenate(out)
    np.save(FEATURES / "question.npy", txt)
    pd.DataFrame({"question_id": qids}).to_csv(FEATURES / "question_index.csv", index=False)
    print(f"question features {txt.shape} in {time.time() - t0:.0f} s")
    assert np.isfinite(img).all() and np.isfinite(txt).all()


if __name__ == "__main__":
    main()
