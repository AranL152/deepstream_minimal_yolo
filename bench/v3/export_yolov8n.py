#!/usr/bin/env python3
"""Export YOLOv8n to two ONNX variants:
  1. yolov8n_ds.onnx  – stock [1, 84, 8400] output for DeepStream's NvDsInferParseYoloCuda
  2. yolov8n_car.onnx – appends top-K + xyxy conversion → [1, 300, 6] for deep_inference.cpp
"""
import shutil
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

def export_stock_onnx(pt_path: Path, out: Path):
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    result = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    exported = Path(result)
    if exported != out:
        shutil.move(str(exported), str(out))
    print(f"Exported stock ONNX → {out}")

def make_car_onnx(stock: Path, out: Path):
    import numpy as np
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh

    m = onnx.load(str(stock))
    g = m.graph
    orig_output_name = g.output[0].name
    g.output.pop()

    def add_init(name, arr):
        g.initializer.append(nh.from_array(np.array(arr), name))

    add_init("perm_s", np.array([0], np.int64))
    add_init("perm_4", np.array([4], np.int64))
    add_init("perm_84", np.array([84], np.int64))
    add_init("ax2", np.array([2], np.int64))
    add_init("half", np.array(0.5, np.float32))
    add_init("k", np.array([300], np.int64))

    nodes = [
        h.make_node("Transpose", [orig_output_name], ["t"], perm=[0, 2, 1]),
        h.make_node("Slice", ["t", "perm_s", "perm_4", "ax2"], ["box"]),
        h.make_node("Slice", ["t", "perm_4", "perm_84", "ax2"], ["cls"]),
        h.make_node("Split", ["box"], ["cxy", "wh"], axis=2, split=[2, 2]),
        h.make_node("Mul", ["wh", "half"], ["hwh"]),
        h.make_node("Sub", ["cxy", "hwh"], ["xy1"]),
        h.make_node("Add", ["cxy", "hwh"], ["xy2"]),
        h.make_node("ReduceMax", ["cls"], ["score"], axes=[2], keepdims=1),
        h.make_node("ArgMax", ["cls"], ["lab_i"], axis=2, keepdims=1),
        h.make_node("Cast", ["lab_i"], ["lab"], to=T.FLOAT),
        h.make_node("Concat", ["xy1", "xy2", "score", "lab"], ["dets"], axis=2),
        h.make_node("Squeeze", ["dets"], ["dets2"], axes=[0]),
        h.make_node("Squeeze", ["score"], ["score1"], axes=[0, 2]),
        h.make_node("TopK", ["score1", "k"], ["topv", "topi"], axis=0),
        h.make_node("Gather", ["dets2", "topi"], ["top"], axis=0),
        h.make_node("Unsqueeze", ["top"], ["output"], axes=[0]),
    ]
    g.node.extend(nodes)
    g.output.append(h.make_tensor_value_info("output", T.FLOAT, [1, 300, 6]))
    onnx.checker.check_model(m)
    onnx.save(m, str(out))
    print(f"Car-format ONNX → {out}")

if __name__ == "__main__":
    pt = DIR.parent.parent / "DeepStream-Yolo" / "yolov8n.pt"
    if not pt.exists():
        sys.exit(f"ERROR: {pt} not found")

    stock = DIR / "yolov8n_ds.onnx"
    car = DIR / "yolov8n_car.onnx"

    if not stock.exists():
        export_stock_onnx(pt, stock)
    else:
        print(f"Stock ONNX already exists: {stock}")

    if not car.exists():
        make_car_onnx(stock, car)
    else:
        print(f"Car ONNX already exists: {car}")
