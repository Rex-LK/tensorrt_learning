from onnxsim import simplify
import onnx
input_path="centernet.onnx"
output_path="centernet_simp.onnx"
onnx_model = onnx.load(input_path)
model_simp, check = simplify(onnx_model,input_shapes={'image': [1, 3, 512, 512]})
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')