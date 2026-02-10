import argparse
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

from datasets import DataConfig, make_dataloaders
from utils.metrics import compute_metrics

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    dcfg = DataConfig(name=args.dataset, img_size=args.img_size, batch_size=args.batch_size)
    _, val_loader, num_classes, _ = make_dataloaders(dcfg)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    y_true, y_pred = [], []
    for x, y in tqdm(val_loader, desc="onnx-eval", leave=False):
        x_np = x.numpy().astype(np.float32)
        logits = sess.run(None, {in_name: x_np})[0]
        pred = logits.argmax(axis=1)
        y_true.append(y.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    class_names = [str(i) for i in range(num_classes)]
    df_class, df_global, _ = compute_metrics(y_true, y_pred, class_names)

    acc = df_global.loc["accuracy", "value"]
    macro_f1 = df_global.loc["macro", "f1"]
    weighted_f1 = df_global.loc["weighted", "f1"]

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

if __name__ == "__main__":
    main()
