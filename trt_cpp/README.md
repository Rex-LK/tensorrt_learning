

#### 1、demo工程

可以先参考demo的里面的工程，下载相应的模型权重，或者直接从给出的百度网盘下载。

#### 2、export_onnx

下载好模型之后，在onnx2trt/demo里面一般可以使用export_onnx.py或者predict.py产生相应的onnx模型，并可以使用infer-onnxruntime使用onnx进行推理，也可以使用onnx_simplify对onnx进行简化。

#### 3、tensorrt

在workspace下面放入需要转化的onnx模型，如vit.onnx、centernet.onnx等

在Makefile中修改本机对应的环境，可以参考手写AI提供的自动环境配置进行，或者修改CMakeLists.txt中本机环境

在src/main.cpp中提供了build_engine 和 inference的代码。直接在onnx2trt文件夹下，进行执行 make run，即可进行推理。

或者在CMakeLists中修改本机的tensorrt、protobuf路径，==可以将模型路径和图片路径修改为绝对路径，执行吗，修改src/main.cpp中的demo名字==

mkdir build && cd build

cmake .. && make -j

在worksapce中执行可执行文件./demo_infer 即可运行相应的demo

#### 4、总结

本次学习课程学习了torch2onnx2trt的方式，同时复习了深度学习中的比较经典的模型，同时对一些模型的后处理采用c++的处理方式，同时也提高了c++的代码能力。



