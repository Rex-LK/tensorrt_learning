import tensorrt as trt

# 构建logger,builder,network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
builder.max_batch_size = 1

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
# 读入onnx查看有无错误
success = parser.parse_from_file("/home/rex/Desktop/tensorrt_learning/trt_py/build_engine/batch_dynamic/detr_sim.onnx")
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if success:
    print('Construction sucess!!!')
    pass  # Error handling code here

profile = builder.create_optimization_profile()
# profile.set_shape("images", (1, 3, 640, 640), (8, 3, 640, 640), (16, 3, 640, 640))
profile.set_shape("images", (1, 3, 800, 1066),(8, 3, 800, 1066), (16, 3,800, 1066))

# profile = builder.create_optimization_profile()
# profile.set_shape("foo", (1,3, 512, 512), (20,3,640, 640), (10,3,640, 640))
config = builder.create_builder_config()
config.add_optimization_profile(profile)
config.max_workspace_size = 1 << 30  #
serialized_engine = builder.build_serialized_network(network, config)
with open("/home/rex/Desktop/tensorrt_learning/trt_py/build_engine/batch_dynamic/hrnet.pyengine", "wb") as f:
    print('正在写入engine文件．．．')
    f.write(serialized_engine)
    print('构建引擎成功！！！')