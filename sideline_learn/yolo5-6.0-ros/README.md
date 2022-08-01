### 【CV学习笔记】ros之yolov5&tensorRT封装调用

#### 1、前言

在yolov5&tensorRT在ros框架下使用时，遇到了一些问题，经过一步一步的排错之后，最终在ros中实现，因此在这里实现了一个简单的demo，便于以后进行回顾，同时给有需要的同学提供一个例子。

yolov5&tensorRT封装以及加速见:[yolov5&tensorRT](https://blog.csdn.net/weixin_42108183/article/details/122352007?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165914415816782184647973%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165914415816782184647973&biz_id=&utm_medium=distribute.pc_search_result.none-task-code-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-122352007-0-null-null.142^v35^experiment_2_v1&utm_term=%E3%80%90CV%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%E3%80%91%20yolov5-6.0%26tensorRT)

代码仓库:https://github.com/Rex-LK/tensorrt_learning/tree/main/side_line_learn/yolo5-6.0-ros

#### 2、编译

2.1、升级cmake

安装ros后，ros中自带某个低版本的cmake，编译会报错，因此需要升级cmake，可以到官网下载cmake-××.sh文件，按照下面的操作步骤更新cmake

```
sudo sh cmake-××.sh
#PATH=/home/rex/cmake-3.22.0-linux-x86_64/bin 为上一步cmake的安装路径
sudo sed -i '$a export PATH=/home/rex/cmake-3.22.0-linux-x86_64/bin:$PATH' ~/.bashrc 
source ~/.bashrc
```

2.2、修改cmakelists

yolo5-6.0-ros/src/yolov5-server/CMakeLists.txt

yolo5-6.0-ros/src/yolov5-infer/CMakeLists.txt

```
#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#tensorRT
include_directories(/home/rex/TensorRT-8.2.0.6/include)
link_directories(/home/rex/TensorRT-8.2.0.6/lib)

# 设置yolov5 项目路径
set(yolov5_dir /home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/src/yolov5-6.0)
```

2.3、编译命令

在yolo5-6.0-ros文件夹中

```
catkin_make_isolated -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel_isolated/setup.bash
```

#### 3、使用单个节点进行推理

```
roslaunch yolov5_infer_nodelet yolov5_infer_nodelet.launch
```

其中launch/yolov5_infer_nodelet.launch文件可以启动对应的节点，并且可以给节点传递参数

```
<?xml version="1.0"?>
<launch>
	<arg name="nodelet_manager" default="yolov5_infer_manager" />
	<node pkg="nodelet" type="nodelet" name="$(arg nodelet_manager)" args="manager" output="screen" />
	//需要启动的节点
	<node pkg="nodelet" type="nodelet" name="yolov5_infer_nodelet" args="standalone yolov5_infer_nodelet/Yolov5InferNodelet" output="screen">
	 //下面为传参
	  <param name="gpu_id" value="0" type="str" />
	  //模型路径
	  <param name="engine_path" value="/home/rex/Desktop/cv_demo/yolov5-6.0-TensorRT/tensorrtx/yolov5s.engine" type="str" />
	  //图片路径
	  <param name="image_path" value="/home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/src/yolov5-6.0/images/zidane.jpg" type="str" />
  	</node>
</launch>
```

#### 4、 使用client/server的方式进行推理

启动server端，server端会等待client端发送请求，并进行推理，返回一个string类型的结果，可以将推理结果转为string后并返回

```
roslaunch yolov5_server_nodelet yolov5_server_nodelet.launch
```

启动client端，client端使用opencv读取一张图片，并转化为ros_images的格式发送给server，得到string类型的结果

```
roslaunch client_nodelet client_nodelet.launch
```

自定义消息格式

```
#client
sensor_msgs/Image image
---
#server
string result
```

#### 5、测试

<img src="Screenshot%20from%202022-07-30%2009-39-36.png" alt="Screenshot from 2022-07-30 09-39-36" style="zoom:50%;" />

#### 6、总结

通过本次demo的实现，学习了在ros中使用nodelet的方式构建并启动一个节点，实现了cliet/server的调用方式，自定义srv消息，并调用yolov5&tensorRT进行推理。