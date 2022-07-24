import sys
sys.path.append("/home/rex/Desktop/tensorrt_learning/trt_py/")
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import numpy as np


def filter_boxes(pred, threshold):
    result = pred.copy()
    result[..., :2] = result[..., :2] - result[..., 2:4] * 0.5
    result[..., 2:4] = result[..., :2] + result[..., 2:4]
    result_selected = result[np.where(result[..., 4] > threshold)]
    boxes = result_selected[..., :4]
    classes = np.argmax(result_selected[..., 5:], axis=-1)
    confs = np.max(result_selected[..., 5:], axis=-1)  # [...,classes]
    # print(boxes.shape)
    # print(classes.shape)
    # print(confs.shape)
    return boxes, confs, classes


def non_max_suppression(boxes, confs, classes, iou_thres=0.6):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    return boxes, confs, classes


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def pre_process(img):
    print('original image shape', img.shape)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.transpose((2, 0, 1)).astype(np.float16)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def allocate_buffers(engine, max_batch_size=16):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        # print(dims)
        if dims[0] == -1:
            assert (max_batch_size is not None)
            dims[0] = max_batch_size  # 动态batch_size适应

        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # print(dtype,size)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)  # 开辟出一片显存
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


with open("yolov5.engine", "rb") as f:
    serialized_engine = f.read()
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
BATCH_SIZE = 8
context = engine.create_execution_context()
context.set_binding_shape(0, (BATCH_SIZE, 3, 640, 640))  # 这句非常重要！！！定义batch为动态维度
inputs, outputs, bindings, stream = allocate_buffers(engine, max_batch_size=BATCH_SIZE)  # 构建输入，输出，流指针

img = cv2.imread('images/street.jpg')
batch_data = np.repeat(pre_process(img),BATCH_SIZE,0)

np.copyto(inputs[0].host, batch_data.ravel())
result = do_inference_v2(context, bindings, inputs, outputs, stream)[0]
result = np.reshape(result,[BATCH_SIZE,-1,85])
print(result.shape)

result = result[0]
img = cv2.resize(img,(640,640))
boxes, confs, classes = filter_boxes(result,0.5)
boxes, confs, classes = non_max_suppression(boxes, confs, classes)
for box,conf,cls in zip(boxes,confs,classes):
    x1,y1,x2,y2 = np.int32(box)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imwrite('tmp.jpg',img)