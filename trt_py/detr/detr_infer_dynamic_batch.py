import os
import cv2
from basic_infer.infer import Infer_bacis, image_resize_pro
import torch
import numpy as np

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

if __name__ == '__main__':
    # batch-infer
    # 假定一个文件夹内的图片为一个batch
    images_dir = 'images/'
    images = os.listdir(images_dir)
    batch_size = len(images)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    image_d_size = (1066, 800)

    batch_image_o = []
    batch_image_o_h = []
    batch_image_o_w = []
    batch_input_image = np.empty(shape=[2, 3, 800, 1066])
    for i, image in enumerate(images):
        img_o = cv2.imread(images_dir + image)
        batch_image_o.append(img_o)
        batch_image_o_h.append(img_o.shape[0])
        batch_image_o_w.append(img_o.shape[1])
        img = image_resize_pro(img_o, image_d_size, imagenet_mean, imagenet_std)
        img = np.ascontiguousarray(img)
        np.copyto(batch_input_image[i], img)

    detr_engine_path = 'detr.pyengine'
    detr_infer = Infer_bacis(detr_engine_path, 2)

    pred = detr_infer.detect(batch_input_image)
    pred = torch.from_numpy(pred.reshape(2, 100, 95))
    print(pred)
    for batch in range(batch_size):
        scores = pred[batch][:, 0:91]
        bboxes = pred[batch][:, 91:]
        keep = scores.max(-1).values > 0.7
        idx = torch.nonzero(keep == True).squeeze()
        scores = scores[keep]
        bboxes = bboxes[keep]
        k = keep.numpy()
        for (score, bbox) in zip(scores, bboxes):
            p_x1 = int((bbox[0] - bbox[2] / 2) * batch_image_o_w[batch])
            p_x2 = int((bbox[0] + bbox[2] / 2) * batch_image_o_w[batch])
            p_y1 = int((bbox[1] - bbox[3] / 2) * batch_image_o_h[batch])
            p_y2 = int((bbox[1] + bbox[3] / 2) * batch_image_o_h[batch])
            p = score.argmax()
            cv2.rectangle(batch_image_o[batch], (p_x1, p_y1), (p_x2, p_y2), (0, 0, 255), 2)
            cv2.putText(batch_image_o[batch], CLASSES[p], (p_x1, p_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image_" + str(batch), batch_image_o[batch])
        cv2.waitKey(0)

    detr_infer.destroy()
