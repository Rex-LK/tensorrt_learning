
import imghdr
import onnxruntime
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch
from model import *
from tools import *


if __name__ == "__main__":

    data_transform = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = "demo.jpg"
    img_o = Image.open(img_path)
    img = data_transform(img_o).unsqueeze(0)
    
    image_input = img.numpy()


    session = onnxruntime.InferenceSession("detr_sim.onnx", providers=["CPUExecutionProvider"])
    pred = session.run(["predict"], {"image": image_input})[0]
    scores = torch.from_numpy(pred[0][:,0:91])
    bboxes = torch.from_numpy(pred[0][:,91:])
    keep = scores.max(-1).values > 0.7
    scores = scores[keep]
    bboxes = bboxes[keep]
    print(bboxes)
    fin_bboxes = rescale_bboxes(bboxes, img_o.size)
    plot_results(img_o, scores, fin_bboxes)
