
from pathlib import Path
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov5n.onnx"

print("\n========== YOLOV5N MODEL PROBE ==========\n")

# Load model
model = onnx.load(str(MODEL_PATH))
graph = model.graph

# Inspect graph operations
print("GRAPH OPERATIONS:")
print(Counter(node.op_type for node in graph.node))

# Find output producer
output_name = graph.output[0].name

print("\nOUTPUT:", output_name)

for node in graph.node:
    if output_name in node.output:
        print("Output producer:", node.op_type)
        print("Inputs:", list(node.input))

# Initialize inference
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]

print("\nINPUT:", input_info.name)
print("INPUT TYPE:", input_info.type)
print("INPUT SHAPE:", input_info.shape)

# Test inference using a zero-filled image tensor
test_input = np.zeros(
    (1, 3, 640, 640),
    dtype=np.float16
)

print("\nRunning test inference...")

outputs = session.run(
    None,
    {input_info.name: test_input}
)

prediction = outputs[0]

print("\nINFERENCE SUCCESSFUL")
print("Output shape:", prediction.shape)
print("Output dtype:", prediction.dtype)
print("Minimum:", float(prediction.min()))
print("Maximum:", float(prediction.max()))
print("Mean:", float(prediction.mean()))

print("\nFIRST 10 CANDIDATES:")
print(prediction[0, :10, :])

print("\n========== PROBE COMPLETE ==========\n")
