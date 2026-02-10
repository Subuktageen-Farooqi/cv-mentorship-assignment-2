import os
import argparse
import numpy as np
from typing import Iterator, Dict

from datasets import DataConfig, make_dataloaders

from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
)

class LoaderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, loader, input_name: str, num_batches: int = 10):
        self.loader = loader
        self.input_name = input_name
        self.num_batches = num_batches
        self._iter = None
        self._count = 0

    def get_next(self) -> Dict[str, np.ndarray]:
        if self._iter is None:
            self._iter = iter(self.loader)

        if self._count >= self.num_batches:
            return None

        x, _ = next(self._iter)
        self._count += 1
        print(f"Calib batch {self._count}/{self.num_batches}", flush=True)
        return {self.input_name: x.numpy().astype(np.float32)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_onnx", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["dynamic", "static"], required=True)

    # for static calibration
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--calib_batches", type=int, default=10)
    ap.add_argument("--input_name", type=str, default="input")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=args.fp32_onnx,
            model_output=args.out,
            weight_type=QuantType.QInt8,
            # adding here 
            op_types_to_quantize=["MatMul","Gemm"],
        )
        print("Saved dynamic INT8:", args.out)
        return

    # static
    dcfg = DataConfig(name=args.dataset, img_size=args.img_size, batch_size=args.batch_size)
    train_loader, _, _, _ = make_dataloaders(dcfg)

    reader = LoaderCalibrationDataReader(train_loader, input_name=args.input_name, num_batches=args.calib_batches)

    quantize_static(
        model_input=args.fp32_onnx,
        model_output=args.out,
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print("Saved static INT8:", args.out)

if __name__ == "__main__":
    main()
