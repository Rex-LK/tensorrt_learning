# from json import tool
# import requests
# import matplotlib.pyplot as plt

# from torchvision.models import resnet50
# import torchvision.transforms as T

# from tools import * 
# from model import *


# if __name__ == '__main__':
#     detr = DETRdemo(num_classes=91)
#     state_dict = torch.load('detr_demo.pth') 
#     detr.load_state_dict(state_dict)
#     detr.eval()

#     dummy = torch.zeros(1,3,800,1066)
#     torch.onnx.export(
#         detr,(dummy,),
#         "detr.onnx",
#         input_names=["image"],
#         output_names=["predict"],
#         opset_version=12,
#         dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"},} 
#     )

# 为了正确导出onnx需要将models/yolo.py文件中56行

# ```python
# # x = x if self.training else (torch.cat(z, 1),x) # 修改为
# x = x if self.training else (torch.cat(z, 1))
# ```

# 在detect.py中采用如下方式导出

# ```
#     dummy = torch.zeros(1,3,640,640)
#     torch.onnx.export(
#         model,(dummy,),
#         "yolov7.onnx",
#         input_names=["image"],
#         output_names=["predict"],
#         opset_version=13,
#         dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"},} 
#     )
# ```