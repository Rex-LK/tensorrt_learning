import cv2
from basic_infer.infer import Infer_bacis, image_resize_pro
import torch
from torch import nn
import numpy as np
from torchvision.ops import nms

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def nms(boxes, scores, iou_threshold):
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    idxs = scores.argsort()  # 值从小到大的 索引
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = idxs.new(keep)  # Tensor
    return keep




if __name__ == '__main__':
    img_path = "images/street.jpg"
    image_o = cv2.imread(img_path)
    image_o_size = (image_o.shape[0], image_o.shape[1])  # h*w
    imagenet_mean = [0.40789655, 0.44719303, 0.47026116]
    imagenet_std = [0.2886383, 0.27408165, 0.27809834]
    image_d_size = (512, 512)
    image_input = image_resize_pro(image_o, image_d_size, imagenet_mean, imagenet_std)

    detr_engine_path = 'centernet.pyengine'
    detr_infer = Infer_bacis(detr_engine_path, batch_size=1)
    pred = detr_infer.detect(image_input)
    batch_size = detr_infer.batch_size
    pred = torch.from_numpy(pred.reshape(batch_size, 128 * 128, 24))[0]

    pred_hms = pred[:, 0:20]
    pred_whs = pred[:, 20:22]
    pred_xys = pred[:, 22:]
    print(pred_hms.shape)
    keep = pred_hms.max(-1).values > 0.3
    hms = pred_hms[keep]
    whs = pred_whs[keep]
    xys = pred_xys[keep]
    scores, labels = torch.max(hms, 1)
    idx = torch.nonzero(keep == True).squeeze()
    score = hms.argmax()
    xys[:, 0] += idx % 128
    xys[:, 1] += idx / 128
    left = (xys[:, 0] - whs[:, 0] / 2) * image_o_size[1] / 128
    print(left)
    top = (xys[:, 1] - whs[:, 1] / 2) * image_o_size[0] / 128
    right = (xys[:, 0] + whs[:, 0] / 2) * image_o_size[1] / 128
    bottom = (xys[:, 1] + whs[:, 1] / 2) * image_o_size[0] / 128
    bboxs = torch.stack((left, top, right, bottom), dim=1)
    nms_keep = nms(bboxs, scores, iou_threshold=0.5)
    bboxs = bboxs[nms_keep]
    # print(scores)
    for bbox, label, score in zip(bboxs, labels, scores):
        cv2.rectangle(image_o, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(image_o, CLASSES[label] + ':' + str(float(score))[0:4], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)
    cv2.imshow("image", image_o)
    cv2.waitKey(0)
