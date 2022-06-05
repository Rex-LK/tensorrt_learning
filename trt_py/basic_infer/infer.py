from cuda import cudart
import tensorrt as trt
import numpy as np
import cv2
import cupy as cp
import torch
from basic_infer import transforms

def inv_mat(M):
    k = M[0, 0],
    b1 = M[0, 2]
    b2 = M[1, 2],
    return np.array([[1 / k[0], 0, -b1 / k[0]],
                     [0, 1 / k[0], -b2[0] / k[0]]])


def image_transfer(image, dst_size):
    oh, ow = image.shape[:2]
    dh, dw = dst_size
    scale = min(dw / ow, dh / oh)

    M = np.array([
        [scale, 0, -scale * ow * 0.5 + dw * 0.5],
        [0, scale, -scale * oh * 0.5 + dh * 0.5]
    ])
    return cv2.warpAffine(image, M, dst_size), M, inv_mat(M)


class Infer_bacis():
    def __init__(self, engine_file_path, batch_size):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        _, self.stream = cudart.cudaStreamCreate()
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.batch_size = batch_size
        # self.context.set_binding_shape(0, (self.batch_size, 3, 800, 1066))
        # assert self.batch_size <= engine.max_batch_size
        for binding in engine:
            size = abs(trt.volume(engine.get_binding_shape(binding))) * self.batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = np.empty(size, dtype=dtype)
            # 19000
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

        # 从这里可以看出 batch推理有问题,只有第一张图片有结果
        # image[0] = image[1]
        batch_input_image = np.ascontiguousarray(image)
        np.copyto(self.host_inputs[0], batch_input_image.ravel())
        cudart.cudaMemcpyAsync(self.cuda_inputs[0], self.host_inputs[0].ctypes.data, self.host_inputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)

        cudart.cudaMemcpyAsync(self.host_outputs[0].ctypes.data, self.cuda_outputs[0], self.host_outputs[0].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        output = self.host_outputs[0]
        return output

    def destroy(self):
        cudart.cudaStreamDestroy(self.stream)
        cudart.cudaFree(self.cuda_inputs[0])
        cudart.cudaFree(self.cuda_outputs[0])


# 1、还需对batch的图片进行推理
# 2、需要记录原图的大小,以及原图，便于后处理


def image_resize_pro(image_o, image_d_size, imagenet_mean, imagenet_std):
    image_input = cv2.resize(image_o, image_d_size)  # resize
    image_input = image_input[..., ::-1]  # BGR -> RGB
    image_input = (image_input / 255.0 - imagenet_mean) / imagenet_std  # normalize
    image_input = image_input.astype(np.float32)  # float64 -> float32
    image_input = image_input.transpose(2, 0, 1)  # HWC -> CHW
    image_input = np.ascontiguousarray(image_input)  # contiguous array memory
    image_input = image_input[None, ...]  # CHW -> 1CHW
    return image_input


def image_warpaffine_pro(image_o, image_d_size, imagenet_mean, imagenet_std):
    img_d, M, inv = image_transfer(image_o, image_d_size)
    # 加速
    image_input = img_d[..., ::-1]  # BGR -> RGB
    image_input = image_input / 255.0
    image_input = (image_input - imagenet_mean) / imagenet_std  # normalize
    image_input = image_input.astype(np.float32)  # float64 -> float32
    image_input = image_input.transpose(2, 0, 1)  # HWC -> CHW
    image_input = image_input[None, ...]  # CHW -> 1CHW

    return image_input, M, inv


class image_torchvision_pro():
    def __init__(self, image_d_size, imagenet_mean, imagenet_std):
        self.image_transform = transforms.Compose([
            transforms.AffineTransform(scale=(1, 1), fixed_size=image_d_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def pro(self, image):
        image_input, target = self.image_transform(image, {"box": [0, 0, image.shape[1] - 1, image.shape[0] - 1]})
        image_input = torch.unsqueeze(image_input, dim=0)
        return image_input, target
