
from pathlib import Path

import numpy as np
import onnxruntime as ort


class YOLOv5Model:

    def __init__(self, model_path, threads=2):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}"
            )

        # Configure ONNX Runtime.
        options = ort.SessionOptions()

        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Load model using CPU.
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        # Inspect model tensors.
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

        expected_input_shape = [1, 3, 640, 640]
        expected_output_shape = [1, 25200, 85]

        if self.input.shape != expected_input_shape:
            raise ValueError(
                f"Unexpected input shape: {self.input.shape}"
            )

        if self.input.type != "tensor(float16)":
            raise ValueError(
                f"Unexpected input type: {self.input.type}"
            )

        if self.output.shape != expected_output_shape:
            raise ValueError(
                f"Unexpected output shape: {self.output.shape}"
            )

        if self.output.type != "tensor(float16)":
            raise ValueError(
                f"Unexpected output type: {self.output.type}"
            )

        print("\n[MODEL] YOLOv5n loaded successfully")
        print("[MODEL] Provider:", self.session.get_providers())
        print("[MODEL] Input:", self.input.shape)
        print("[MODEL] Output:", self.output.shape)

    def infer(self, tensor):

        if tensor.shape != (1, 3, 640, 640):
            raise ValueError(
                f"Invalid input tensor shape: {tensor.shape}"
            )

        if tensor.dtype != np.float16:
            raise ValueError(
                f"Expected float16 tensor, received {tensor.dtype}"
            )

        result = self.session.run(
            [self.output.name],
            {self.input.name: tensor}
        )

        return result[0]
