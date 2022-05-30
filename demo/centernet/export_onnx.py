from torch import nn
import torch
from nets.centernet import CenterNet_Resnet50
import cv2
import numpy as np
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from PIL import Image

from utils.utils_bbox import  postprocess


class myCenternet(nn.Module):
    def __init__(self):
        super(myCenternet, self).__init__()
        self.model_path = 'centernet_resnet50_voc.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = CenterNet_Resnet50(num_classes=20, pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval()
        self.num_class = 20
        self.feature_map = 128

    def pool_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep


    def forward(self, x):
        x = self.model(x)
        hms = self.pool_nms(x[0]).view(1, self.num_class, self.feature_map * self.feature_map).permute(0, 2, 1)
        whs = x[1].view(1, 2, self.feature_map * self.feature_map).permute(0, 2, 1)
        xys = x[2].view(1, 2, self.feature_map * self.feature_map).permute(0, 2, 1)
        y = torch.cat((hms,whs),dim=2)
        y = torch.cat((y,xys),dim=2)
        return y


model = myCenternet()
dummy = torch.zeros(1, 3, 512, 512)
torch.onnx.export(
    model, (dummy,),
    "centernet.onnx",
    input_names=["image"],
    output_names=["predict"],
    opset_version=13,
    dynamic_axes={"image": {0: "batch"}}
)
