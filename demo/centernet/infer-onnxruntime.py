from tkinter import image_names
import onnxruntime
import cv2
import numpy as np
from PIL import  Image
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import torch
image = 'img/street.jpg'
image = Image.open(image)
image_shape = np.array(np.shape(image)[0:2])
image = cvtColor(image)
image_data  = resize_image(image, (512,512), None)
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
images = np.asarray(image_data).astype(np.float32)
session = onnxruntime.InferenceSession("centernet.onnx", providers=["CPUExecutionProvider"])
pred = session.run(["predict"], {"image": images})[0]
pred = torch.from_numpy(pred)
pred_hms = pred[0,:,0:20]
pred_wh =  pred[0,:,20:22]
pred_xy =  pred[0,:,22:]
keep = pred_hms.max(-1).values > 0.3
pred_hms = pred_hms[keep]
pred_wh =  pred_wh[keep]
pred_xy =  pred_xy[keep]
print(pred_wh.shape)