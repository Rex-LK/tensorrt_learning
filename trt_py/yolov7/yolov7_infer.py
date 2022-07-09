import cv2
import sys

sys.path.append("/home/rex/Desktop/tensorrt_learning/trt_py")
from basic_infer.infer import Infer_bacis, image_warpaffine_pro
from util.util import *
import torch
import numpy as np

if __name__ == '__main__':
    img_path = "images/street.jpg"
    image_o = cv2.imread(img_path)
    image_o_size = (image_o.shape[0], image_o.shape[1])  # h*w
    img, _, inv = image_warpaffine_pro(image_o, (640, 640))
    print(inv )
    yolov7_engine_path = 'yolov7.pyengine'
    detr_infer = Infer_bacis(yolov7_engine_path, batch_size=1)
    pred = detr_infer.detect(img)
    batch_size = detr_infer.batch_size
    pred = torch.from_numpy(pred.reshape(batch_size, -1, 85))
    for bt in range(batch_size):
        cur_pred = pred[bt]
        keep_objectness = cur_pred[:, 4] > 0.25
        cur_pred = cur_pred[keep_objectness]
        scores_prob, label = torch.max(cur_pred[:, 4:], 1)
        cur_pred[:, 4] *= scores_prob
        keep_scores = cur_pred[:, 4] > 0.25
        cur_pred = cur_pred[keep_scores]
        bboxes = xywh2xyxy(cur_pred[:, :4])
        scores = cur_pred[:, 4]
        keep_nms = nms(bboxes, cur_pred[:, 4], 0.35)

        bboxes = bboxes[keep_nms]
        scores = scores[keep_nms]
        if len(bboxes):
            bboxes[:, 0] = bboxes[:, 0] * inv[0][0] + inv[0][2]
            bboxes[:, 1] = bboxes[:, 1] * inv[1][1] + inv[1][2]
            bboxes[:, 2] = bboxes[:, 2] * inv[0][0] + inv[0][2]
            bboxes[:, 3] = bboxes[:, 3] * inv[1][1] + inv[1][2]
            for bbox, score in zip(bboxes, scores):
                cv2.rectangle(image_o, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.imshow("image_o", image_o)
            cv2.waitKey(0)
        # if len(bboxes):
        #     for bbox, score in zip(bboxes, scores):
        #         cv2.rectangle(image_o, (int(bbox[0]*2.078125+0.5390625), int(bbox[1]*2.078125+0.5390625)), (int(bbox[2]*2.078125+0.5390625), int(bbox[3]*2.078125+0.5390625)), (0, 255, 0), 2)
        #     cv2.imshow("image_o", image_o)
        #     cv2.waitKey(0)
