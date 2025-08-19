#!/usr/bin/env python3

import os
import sys
from pathlib import Path

from ultralytics import YOLO
import onnx

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
model_path = project_root / "models" / "yolo11n.pt"
output_dir = project_root / "models"
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)
print(f"Loaded model from {model_path}")
onnx_path = model.export(format="onnx", imgsz=640)
print(f"Exported model to {output_dir}")

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("Verified")

# Run inference
onnx_model_path = output_dir / "yolo11n.onnx"
onnx_model = YOLO(onnx_model_path)
results = onnx_model("https://ultralytics.com/images/bus.jpg")
print(f"inference results: {results}")
