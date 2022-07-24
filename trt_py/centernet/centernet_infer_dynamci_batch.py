import sys
sys.path.append("/home/rex/Desktop/tensorrt_learning/trt_py")
print(sys.path)
import cv2
from basic_infer.infer import Infer_bacis, image_resize_pro
import torch
from torch import nn
import numpy as np
import os

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


def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


if __name__ == '__main__':
    # batch-infer
    # 假定一个文件夹内的图片为一个batch
    images_dir = '/home/rex/Desktop/tensorrt_learning/trt_py/centernet/images/'
    images = os.listdir(images_dir)
    batch_size = len(images)

    batch_image_o = []
    batch_image_o_h = []
    batch_image_o_w = []
    batch_input_image = np.empty(shape=[batch_size, 3, 512, 512])

    imagenet_mean = [0.40789655, 0.44719303, 0.47026116]
    imagenet_std = [0.2886383, 0.27408165, 0.27809834]
    image_d_size = (512, 512)
    for i, image in enumerate(images):
        img_o = cv2.imread(images_dir + image)
        batch_image_o.append(img_o)
        batch_image_o_h.append(img_o.shape[0])
        batch_image_o_w.append(img_o.shape[1])
        img = image_resize_pro(img_o, image_d_size, imagenet_mean, imagenet_std)
        np.copyto(batch_input_image[i], img)

    detr_engine_path = 'centernet/centernet.pyengine'
    detr_infer = Infer_bacis(detr_engine_path, batch_size=2)
    pred = detr_infer.detect(batch_input_image)

    batch_size = detr_infer.batch_size
    pred = torch.from_numpy(pred.reshape(batch_size, 128 * 128, 24))[1]

    print(pred)



    # pred_hms = pred[:, 0:20]
    # pred_whs = pred[:, 20:22]
    # pred_xys = pred[:, 22:]
    #
    # keep = pred_hms.max(-1).values > 0.3
    #
    # hms = pred_hms[keep]
    # whs = pred_whs[keep]
    # xys = pred_xys[keep]
    # scores, labels = torch.max(hms, 1)
    # # 在一维数组中找到指定元素的索引
    # idx = torch.nonzero(keep == True).squeeze()
    # score = hms.argmax()
    # xys[:, 0] += idx % 128
    # xys[:, 1] += idx / 128
    #
    # left = (xys[:, 0] - whs[:, 0] / 2) * image_o_size[1] / 128
    # top = (xys[:, 1] - whs[:, 1] / 2) * image_o_size[0] / 128
    # right = (xys[:, 0] + whs[:, 0] / 2) * image_o_size[1] / 128
    # bottom = (xys[:, 1] + whs[:, 1] / 2) * image_o_size[0] / 128
    # bboxs = torch.stack((left, top, right, bottom), dim=1)
    # nms_keep = nms(bboxs, scores, iou_threshold=0.5)
    # bboxs = bboxs[nms_keep]
    # for bbox, label, score in zip(bboxs, labels, scores):
    #     cv2.rectangle(image_o, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    #     cv2.putText(image_o, CLASSES[label] + ':' + str(float(score))[0:4], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.75, (0, 0, 255), 2)
    # cv2.imshow("image", image_o)
    # cv2.waitKey(0)
