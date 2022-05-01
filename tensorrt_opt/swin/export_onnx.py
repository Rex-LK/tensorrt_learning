
from multiprocessing import dummy
import os
import json
import onnx
from psutil import cpu_count
import sys


import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model


def main():

    model = create_model(num_classes=5).to("cpu")
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    dummy = torch.zeros(1,3,224,224)
    torch.onnx.export(
        model,(dummy,),
        "swin.onnx",
        input_names=["image"],
        output_names=["predict"],
        opset_version=12,
        dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"}} 
    )


if __name__ == '__main__':
    main()
