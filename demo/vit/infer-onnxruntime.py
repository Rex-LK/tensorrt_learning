
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
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "demo.jpg"
    img = Image.open(img_path)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    image_input = img.numpy()
    session = onnxruntime.InferenceSession("vit.onnx", providers=["CPUExecutionProvider"])
    pred = session.run(["predict"], {"image": image_input})[0]
    print(pred)