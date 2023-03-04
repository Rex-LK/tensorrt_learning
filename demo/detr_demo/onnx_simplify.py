from onnxsim import simplify
import onnx
input_path="detr.onnx"
output_path="detr_sim.onnx"
onnx_model = onnx.load(input_path)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')