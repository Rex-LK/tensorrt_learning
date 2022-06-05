import tensorrt as trt
import os
import common

def build_engine(onnx_file_path, engine_file_path, input_shape, TRT_LOGGER, max_batch_size, max_workspace_size):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, builder.create_builder_config() \
            as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = 1 << max_workspace_size  # 256MiB
        builder.max_batch_size = max_batch_size
        # builder.fp16_mode = True
        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)

        if not os.path.exists(onnx_file_path):
            print(f"{onnx_file_path} is not exits")
            exit(0)
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        network.get_input(0).shape = input_shape
        print('Building an engine ... please wait for a while...')
        plan = builder.build_serialized_network(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        print("Completed creating Engine")
        return

if __name__ == '__main__':
    input_shape = [1, 3, 256, 192]
    onnx_path = 'hrnet.onnx'
    pyengine_path = 'hrnet.pyengine'
    max_batch_size = 1
    max_workspace_size = 30 # 1<<28
    TRT_LOGGER = trt.Logger()
    build_engine(onnx_path, pyengine_path, input_shape,TRT_LOGGER , max_batch_size , max_workspace_size)

