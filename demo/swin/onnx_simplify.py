from onnxsim import simplify
import onnx
input_path="swin.onnx"
output_path="swin_sim.onnx"
onnx_model = onnx.load(input_path)
model_simp, check = simplify(onnx_model,input_shapes={'image': [1, 3, 224, 224]},dynamic_input_shape==True)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')