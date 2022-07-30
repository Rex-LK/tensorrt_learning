### CV学习笔记之tensorrt__cuda_python

#### 1、前言

在使用tensorrt的时候，一般是使用cpp，对于cpp基础一般的同学不是很友好，尤其是在学习的过程中，而cpp主要是在部署的时候用到，最近了解到了Nvidia推出的cuda-python库，与之前的pycuda有类似的功能，但整体的编码风格与cpp类似，可以参考下文的代码，转成tensorrt之后，可以在python中先编写后处理的方式，有需要时再改写成cpp，也是一种不错的方式，但python版本的tensorrt相对于cpp来说仍然有不少的局限性。

个人学习代码地址为:

https://github.com/Rex-LK/tensorrt_learning

(ps: 本仓库的代码暂时只支持单张图片推理，由于水平有限，这个问题暂时没有得到解决，要是有哪位大佬知道，麻烦解答一下)

### 2、环境配置

2.1、推荐在conda环境下进行安装，更推荐miniconda

2.2、cuda-python，这个库可以采用pip安装，也可以下载github中对应的仓库代码进行编译安装

2.3、torch，onnx，尽量安装最新的版本,避免出现bug

### 3、build_engine

3.1、exoprt_onnx

通常情况下，项目代码都提供了导出onnx模型的方式，但是通常都不是我们想要的onnx模型，或者这个onnx的模型还能得到一定的优化，比如将部分后处理放在onnx模型，就据需要自定义导出onnx模型了，可以参考https://blog.csdn.net/weixin_42108183/article/details/124969680

3.2、onnx2trt

python版本与cpp版本的onnx2trt的方式十分类似，首先来看cpp是如何来构造引擎的，这是最基本的构造engine的方法，更方便的代码可以参考[手写AI](https://github.com/shouxieai/tensorRT_Pro)

```cpp
bool build_model(){
    TRTLogger logger;
	//构造engine需要下面几个组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("classifier.onnx", 1)){
        printf("Failed to parse classifier.onnx\n");
        return false;
    }
    
    int maxBatchSize = 10;
    //工作空间大小
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("engine.engine", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}
```

从上面的代码可以看出，通常情况下，转onnx需要下面几个组件builder、config、network、parser,接下来看看在python中构造引擎的方式，是不是与cpp的转化方式如出一辙呢。

```python
import tensorrt as trt
import os
import common

def build_engine(onnx_file_path, engine_file_path, input_shape, TRT_LOGGER, max_batch_size, max_workspace_size):
    # builder、network、parser、config在这里体现出来了，与cpp的方式一致
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, builder.create_builder_config() \
            as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = 1 << max_workspace_size  # 256MiB
        builder.max_batch_size = max_batch_size

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
        #设置输入的大小
        network.get_input(0).shape = input_shape
        print('Building an engine ... please wait for a while...')
        plan = builder.build_serialized_network(network, config)
        #储存为文件
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        print("Completed creating Engine")
        return

if __name__ == '__main__':
    input_shape = [1, 3, 512, 512]
    onnx_path = 'centernet.onnx'
    pyengine_path = 'centernet.pyengine'
    max_batch_size = 1
    max_workspace_size = 30 # 1<<28
    TRT_LOGGER = trt.Logger()
    build_engine(onnx_path, pyengine_path, input_shape,TRT_LOGGER , max_batch_size , max_workspace_size)
```

通过上面的代码通常可以将onnx模型转化为engine，但是使用的是trt.OnnxParser，有些模型会转化失败，而使用onnx-tensorrt源码来进行构造引擎时，自由度更高，成功的概率也越大，但是对于一般的学习来说，trt.OnnxParser基本上足够了。

### 4、单张图片推理

整个过程也是分为三个过程，图像预处理、推理、后处理，这里着重比较一下cuda-cpp、cuda-python以及pycuda三者之间的异同，首先查看cpp中在推理前后需要做的事情

```cpp
cudaStream_t stream = nullptr;
checkRuntime(cudaStreamCreate(&stream));
auto execution_context = make_nvshared(engine->createExecutionContext());

int input_batch = 1;
int input_channel = 3;
int input_height = 512;
int input_width = 512;
int input_numel = input_batch * input_channel * input_height * input_width;
float* input_data_host = nullptr;
float* input_data_device = nullptr;
//分配input的Host空间
checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
//分配intput的Device空间
checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

//分配output的空间
float output_data_host[num_classes];
float* output_data_device = nullptr;
//分配output的Device空间
checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

//绑定输入输出
float* bindings[] = {input_data_device, output_data_device};
//推理
bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
//将结果拷贝到Host上
checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
checkRuntime(cudaStreamSynchronize(stream));

```

大致可分为三步，分配空间、推理、复制结果。接下来看看在cuda-python之前是怎么进行推理的，以pycuda为例，该例子可以见https://github.com/NVIDIA/TensorRT/tree/main/samples/python/yolov3_onnx

```python
#分配空间
def allocate_buffers(engine, max_batch_size=16):
    #输入
    inputs = []
    #输出
    outputs = []
    bindings = []
   	#创建流
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        # print(dims)
        if dims[0] == -1:
            assert (max_batch_size is not None)
            dims[0] = max_batch_size  # 动态batch_size适应

        #计算所需空间
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        #分配host空间
        host_mem = cuda.pagelocked_empty(size, dtype)
        #分配device空间
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        #绑定输入输出
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

#推理
def do_inference_v2(context, bindings, inputs, outputs, stream):
 	#复制数据到device
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    #推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    #复制结果到host上
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    return [out.host for out in outputs]
```

最后来看看cuda-python库是怎么操作的，我对cuda-python简单的封装了一个Infer类，在初始化中分配空间，在detect函数中进行推理。

```python
from cuda import cudart
import tensorrt as trt
class Infer_bacis():
    def __init__(self, engine_file_path, batch_size):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        #反序列化engine
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        #构造上下文
        self.context = engine.create_execution_context()
        #创建流
        _, self.stream = cudart.cudaStreamCreate()
        #定义输入输出
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.batch_size = batch_size
        #self.context.set_binding_shape(0, (2, 3, 512, 512))
        # assert self.batch_size <= engine.max_batch_size
        for binding in engine:
            #计算空间大小
            size = abs(trt.volume(engine.get_binding_shape(binding))) * self.batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = np.empty(size, dtype=dtype)
         	#分配空间 
            _, cuda_mem = cudart.cudaMallocAsync(host_mem.nbytes, self.stream)
            self.bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
    def detect(self, image):
        batch_input_image = np.ascontiguousarray(image)
        np.copyto(self.host_inputs[0], batch_input_image.ravel())
        cudart.cudaMemcpyAsync(self.cuda_inputs[0], self.host_inputs[0].ctypes.data, self.host_inputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
		
        #推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)

        cudart.cudaMemcpyAsync(self.host_outputs[0].ctypes.data, self.cuda_outputs[0], self.host_outputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        output = self.host_outputs[0]
        return output
```

从cuda-python的方式来看，其中数据的复制方式与cpp更加类似，函数名也几乎一致，这样用起来也更加方便了。

### 5、总结

本次学习了cuda-python、tensorrt的实现过程，包括torch转onnx、onnx转tensorrt的方法，虽然目前并不完善，后续也会进行相应的跟进，解决多batch推理的问题。同时了解到了在python上进行cuda加速的方法，后续会对cpp和python的tensorrt方法持续跟进。





