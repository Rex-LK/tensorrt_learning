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
        self.model_path = 'model_data/centernet_resnet50_voc.pth'
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

    def decode_bbox(self, pred_hms, pred_whs, pred_offsets, confidence):
        # [1,20,128,128]
        pred_hms = self.pool_nms(pred_hms)

        # b = 2,c = 20 , output_h = 128,output_w=128
        b, c, output_h, output_w = map(int, pred_hms.shape)
        # [20,128,128] -> [128,128,20] ->[128*128, 20]
        heat_map = pred_hms[0].permute(1, 2, 0).view([output_h * output_w, c])
        # [2,128,128] -> [128,128,2] ->[128*128, 2]
        pred_wh = pred_whs[0].permute(1, 2, 0).view([output_h * output_w, 2])
        # [2,128,128] -> [128,128,2] ->[128*128, 2]
        pred_offset = pred_offsets[0].permute(1, 2, 0).view([output_h * output_w, 2])
        prob = torch.cat((heat_map, pred_wh), dim=1).contiguous()
        prob = torch.cat((prob, pred_offset), dim=1).contiguous()
        return prob

        # #两个[128,128]
        # yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        # xv, yv = xv.flatten().float(), yv.flatten().float()
        # # 128*128行，20列 对每一行求最大值
        # class_conf, class_pred = torch.max(heat_map, dim=-1)
        # #[128*128] 找到里面最大的值大于 confidence的为true,否则为false
        # mask = class_conf > confidence
        # #找到大于阈值的点
        # pred_wh_mask = pred_wh[mask]
        # pred_offset_mask = pred_offset[mask]
        # xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        # yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        # half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        # bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        # bboxes[:, [0, 2]] /= output_w
        # bboxes[:, [1, 3]] /= output_h
        # detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],1), torch.unsqueeze(class_pred[mask],1).float()], dim=-1)
        # return detect

    def forward(self, x):
        x = self.model(x)
        hms = self.pool_nms(x[0]).view(1, self.num_class, self.feature_map * self.feature_map).permute(0, 2, 1)
        whs = x[1].view(1, 2, self.feature_map * self.feature_map).permute(0, 2, 1)
        xys = x[2].view(1, 2, self.feature_map * self.feature_map).permute(0, 2, 1)
        y = torch.cat((hms,whs),dim=2)
        y = torch.cat((y,xys),dim=2)
        return y

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


model = myCenternet()

image = 'img/street.jpg'
image = Image.open(image)
image_shape = np.array(np.shape(image)[0:2])
image = cvtColor(image)
image_data = resize_image(image, (512, 512), None)
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
pred = model.forward(images).detach().numpy().astype(np.float32)
save_tensor(pred,"pred.tensor")

# pred_hms = pred[0,:,0:20]
# pred_wh =  pred[0,:,20:22]
# pred_xy =  pred[0,:,22:]
#
# yv, xv = torch.meshgrid(torch.arange(0, 128), torch.arange(0, 128))
# xv, yv      = xv.flatten().float(), yv.flatten().float()
# class_conf, class_pred  = torch.max(pred_hms, dim = -1)
# mask                    = class_conf > 0.3
#
# pred_wh_mask = pred_wh[mask]
# pred_offset_mask = pred_xy[mask]
#
# xv_mask = (xv[mask] + pred_offset_mask[..., 0]).unsqueeze(1)
# yv_mask = (yv[mask] + pred_offset_mask[..., 1]).unsqueeze(1)
# half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
# bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
# bboxes[:, [0, 2]] /= 128
# bboxes[:, [1, 3]] /= 128
# detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()],dim=-1)
# print(detect)
# pred_result = []
# pred_result.append(detect)
# results = postprocess(pred_result, True, image_shape, 512, False, 0.3)

# print(pred.shape)
# dummy = torch.zeros(1, 3, 512, 512)
# torch.onnx.export(
#     model, (dummy,),
#     "centernet.onnx",
#     input_names=["image"],
#     output_names=["predict"],
#     opset_version=12,
#     dynamic_axes={"image": {0: "batch"}}
# )
