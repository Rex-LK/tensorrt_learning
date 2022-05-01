import onnx
import onnxruntime
model = onnx.load("swin.onnx")

model()