
from pathlib import Path
import onnx
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov5n.onnx"

print("\n========== DRONE VISION MODEL INSPECTOR ==========\n")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Model:", MODEL_PATH)
print("File size:", MODEL_PATH.stat().st_size, "bytes")

# Validate ONNX structure
model = onnx.load(str(MODEL_PATH))
onnx.checker.check_model(model)

print("\nONNX validation: PASSED")

# Initialize CPU inference session
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("\nINPUT TENSORS")

for item in session.get_inputs():
    print({
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    })

print("\nOUTPUT TENSORS")

for item in session.get_outputs():
    print({
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    })

print("\nExecution providers:", session.get_providers())

print("\n========== INSPECTION COMPLETE ==========\n")
