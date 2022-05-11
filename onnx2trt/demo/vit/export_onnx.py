
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

from vit_model import vit_base_patch16_224_in21k as create_model


def main():

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = create_model(num_classes=5, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    dummy = torch.zeros(1,3,224,224)
    torch.onnx.export(
        model,(dummy,),
        "vit.onnx",
        input_names=["image"],
        output_names=["predict"],
        opset_version=12,
        dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"}} 
    )


if __name__ == '__main__':
    main()
