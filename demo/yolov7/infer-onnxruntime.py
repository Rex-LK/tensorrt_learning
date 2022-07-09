import transforms
import onnxruntime
import torch
import cv2
import numpy as np

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

def save_tensor(tensor, file):

    with open(file, "wb") as f:
        typeid = 0
        if tensor.dtype == np.float32:
            typeid = 0
        elif tensor.dtype == np.float16:
            typeid = 1
        elif tensor.dtype == np.int32:
            typeid = 2
        elif tensor.dtype == np.uint8:
            typeid = 3

        head = np.array([0xFCCFE2E2, tensor.ndim, typeid], dtype=np.uint32).tobytes()
        f.write(head)
        f.write(np.array(tensor.shape, dtype=np.uint32).tobytes())
        f.write(tensor.tobytes())

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite("2.jpg",img)
    return img, ratio, (dw, dh)


if __name__ == "__main__":

    img_path = "/home/rex/Desktop/tensorrt_learning/demo/yolov7/inference/images/street.jpg"
    img = cv2.imread(img_path)
    img = letterbox(img, (640), 32)[0]
    print(img.shape)
        # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype('float32')
    img  /= 255.0
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    session = onnxruntime.InferenceSession("yolov7.onnx", providers=["CPUExecutionProvider"])
    pred = session.run(["predict"], {"image": img_tensor.numpy()})[0]
    print(pred.shape)
    pred = torch.from_numpy(pred)
    pred = non_max_suppression(pred, 0.25, 0.35)
    print(pred[0].shape)
    # save_tensor(pred.numpy(),"yolov7.tensor")

