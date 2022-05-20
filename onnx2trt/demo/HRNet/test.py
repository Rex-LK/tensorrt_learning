import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms



def predict_single_person():
    resize_hw = (256, 192)
    img_path = "person.png"
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1, 1), fixed_size=resize_hw),
    ])

    # read single-person image
    img = cv2.imread(img_path)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    cv2.imshow("1",img_tensor)
    cv2.waitKey(0)


predict_single_person()