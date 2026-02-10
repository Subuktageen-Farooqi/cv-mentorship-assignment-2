import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from datasets import IMAGENET_MEAN, IMAGENET_STD, DataConfig, make_dataloaders
from models import make_model
from utils.metrics import compute_metrics, save_confusion_matrix

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="part_1_results")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt["arch"]
    dataset = ckpt["dataset"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt.get("img_size", 224)
    pretrained = ckpt.get("pretrained", True)

    # recreate loaders
    dcfg = DataConfig(name=dataset, img_size=img_size, batch_size=256)
    _, val_loader, _, _ = make_dataloaders(dcfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(arch, num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    y_true, y_pred = collect_preds(model, val_loader, device)

    class_names = [str(i) for i in range(num_classes)]
    df_class, df_global, cm = compute_metrics(y_true, y_pred, class_names)

    os.makedirs(args.out_dir, exist_ok=True)
    cm_path = os.path.join(args.out_dir, f"cn_{arch}_{dataset}.png")
    save_confusion_matrix(cm, class_names, cm_path)

    # append to summary.txt in pandas-style tables
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"Experiment: {arch} | Dataset: {dataset}\n")
        f.write(df_class.to_string())
        f.write("\n\n")
        f.write(df_global.to_string())
        f.write("\n\n")

    print("Saved confusion matrix:", cm_path)
    print("Updated summary:", summary_path)

if __name__ == "__main__":
    main()
