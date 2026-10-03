# Append [1,84,8400] -> [1,8400,6] (x1,y1,x2,y2,score,class) to stock yolov8s.onnx,
# top-300 by score, i.e. the output layout deep_inference.cpp's YOLO path reads.
import onnx, numpy as np
from onnx import helper as h, numpy_helper as nh, TensorProto as T
m = onnx.load("../DeepStream-Yolo/yolov8s.onnx")
g = m.graph
g.output.pop()
c = lambda n, a: g.initializer.append(nh.from_array(np.array(a), n))
c("perm_s", np.array([0], np.int64)); c("perm_4", np.array([4], np.int64)); c("perm_84", np.array([84], np.int64))
c("ax2", np.array([2], np.int64)); c("s2", np.array([2, 2], np.int64)); c("half", np.array(0.5, np.float32)); c("k", np.array([300], np.int64))
n = [
  h.make_node("Transpose", ["output0"], ["t"], perm=[0, 2, 1]),               # [1,8400,84]
  h.make_node("Slice", ["t", "perm_s", "perm_4", "ax2"], ["box"]),           # cx,cy,w,h
  h.make_node("Slice", ["t", "perm_4", "perm_84", "ax2"], ["cls"]),          # 80 scores
  h.make_node("Split", ["box"], ["cxy", "wh"], axis=2, split=[2, 2]),
  h.make_node("Mul", ["wh", "half"], ["hwh"]),
  h.make_node("Sub", ["cxy", "hwh"], ["xy1"]),
  h.make_node("Add", ["cxy", "hwh"], ["xy2"]),
  h.make_node("ReduceMax", ["cls"], ["score"], axes=[2], keepdims=1),
  h.make_node("ArgMax", ["cls"], ["lab_i"], axis=2, keepdims=1),
  h.make_node("Cast", ["lab_i"], ["lab"], to=T.FLOAT),
  h.make_node("Concat", ["xy1", "xy2", "score", "lab"], ["dets"], axis=2),  # [1,8400,6]
  # keep top-300 by score, like an end-to-end (YOLOv10/YOLO26-style) export -> [1,300,6]
  h.make_node("Squeeze", ["dets"], ["dets2"], axes=[0]),
  h.make_node("Squeeze", ["score"], ["score1"], axes=[0, 2]),
  h.make_node("TopK", ["score1", "k"], ["topv", "topi"], axis=0),
  h.make_node("Gather", ["dets2", "topi"], ["top"], axis=0),
  h.make_node("Unsqueeze", ["top"], ["output"], axes=[0]),
]
g.node.extend(n)
g.output.append(h.make_tensor_value_info("output", T.FLOAT, [1, 300, 6]))
onnx.checker.check_model(m)
onnx.save(m, "yolov8s_car.onnx")
print("ok")
