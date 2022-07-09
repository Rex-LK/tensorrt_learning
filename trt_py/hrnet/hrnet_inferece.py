import cv2
from basic_infer.infer import Infer_bacis, image_resize_pro, image_warpaffine_pro, image_torchvision_pro
import torch
import numpy as np
import cupy as cp
import time
import sys
sys.path.append("/home/rex/Desktop/tensorrt_learning/trt_py")


def load_tensor(file):
    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    else:
        assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"

    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


if __name__ == '__main__':
    img_path = "images/person.png"
    image_o = cv2.imread(img_path)
    image_o_size = (image_o.shape[0], image_o.shape[1])  # h*w
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_d_size = (256, 192)
    image_pro = image_torchvision_pro(image_d_size, imagenet_mean, imagenet_std)
    image_input, M = image_pro.pro(image_o)
    inv_M = M["reverse_trans"]
    batch_size = 1
    hrnet_engine_path = 'hrnet.pyengine'
    hrnet_infer = Infer_bacis(hrnet_engine_path, batch_size)
    pred = hrnet_infer.detect(image_input)
    use_cupy = 0
    use_torch = not use_cupy
    # 直接在后处理中使用cupy是比较耗时间的,需要从整体上替换成cupy,应该可以减少时间
    if use_cupy:
        print("use_cupy")
        t_start = int(time.time() * 1000)
        pred = cp.asarray(pred)
        t_process_start = int(time.time() * 1000)
        print(f"time in numpy2cupy = {t_process_start - t_start} ms")
        pred = pred.reshape(17, 64 * 48)
        indx = cp.argmax(pred, 1)
        scores = cp.max(pred, axis=1)
        keep = scores > 0.2
        indx = indx[keep]
        scores = scores[keep]
        t_end = int(time.time() * 1000)
        print(f'use_cupy postprocess time : {t_end - t_process_start} ms')


    elif use_torch:
        print("use_torch")
        t_start = int(time.time() * 1000)
        pred = torch.from_numpy(pred)
        t_process_start = int(time.time() * 1000)
        print(f"time in numpy2cupy = {t_process_start - t_start} ms")
        pred = pred.reshape(1, 17, 64 * 48).squeeze(0)
        keep = pred.max(-1).values > 0.2
        pred = pred[keep]
        scores, indx = torch.max(pred, 1)
        t_end = int(time.time() * 1000)
        print(f'use_torch postprocess time : {t_end - t_process_start} ms')

    x = indx % 48
    y = indx / 48
    x_os = (x * inv_M[0][0]) + inv_M[0][2]
    y_os = (y * inv_M[1][1]) + inv_M[1][2]
    for x_o, y_o in zip(x_os, y_os):
        cv2.circle(image_o, (int(x_o), int(y_o)), 1, (255, 0, 0), 2)

    cv2.imshow('image', image_o)
    cv2.imwrite('hrnet-pred.jpg',image_o)
    cv2.waitKey(0)
