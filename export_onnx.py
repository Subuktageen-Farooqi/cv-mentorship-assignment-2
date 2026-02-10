import os
import argparse
import torch

from models import make_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="part_2_results/model_fp32.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt["arch"]
    num_classes = ckpt["num_classes"]
    pretrained = ckpt.get("pretrained", True)
    img_size = ckpt.get("img_size", 224)

    model = make_model(arch, num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset
    )

    print("Exported ONNX:", args.out)

if __name__ == "__main__":
    main()
