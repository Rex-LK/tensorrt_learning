
import onnxruntime
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch


if __name__ == "__main__":

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "flower_photos/test.jpg"
    img = Image.open(img_path)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    image_input = img.numpy()
    session = onnxruntime.InferenceSession("weights/swin.onnx", providers=["CPUExecutionProvider"])
    pred = session.run(["predict"], {"image": image_input})[0]
    print(pred)

#t0.51477224 0.28504044 0.05650776 0.08253327 0.06114639