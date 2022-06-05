import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms

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

def predict_all_person():
    # TODO
    pass


def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = False
    resize_hw = (256, 192)
    img_path = "person2.jpeg"
    weights_path = "./hrnet_w32.pth.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1, 1), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape[1] - 1)
    print(img.shape[0] - 1)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    print(img_tensor.shape)
    print(target["reverse_trans"])
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # torch.onnx.export(
    #     model,(img_tensor,),
    #     "hrnet.onnx",
    #     input_names=["image"],
    #     output_names=["predict"],
    #     opset_version=12,
    #     dynamic_axes={"image":{0:"batch"},"predict":{0:"batch"}}
    # )
    with torch.inference_mode():
        outputs = model(img_tensor.to(device))
        # save_tensor(outputs.cpu1().numpy().astype(np.float32),"hrnet.tensor")
        # 特征图64*48 然后预测17个点(假定一个人上有17个点）
        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5
        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    predict_single_person()
