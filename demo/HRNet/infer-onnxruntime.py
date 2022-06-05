import transforms
import onnxruntime
import torch
import cv2
import numpy as np
if __name__ == "__main__":

    resize_hw = (256, 192)
    img_path = "person.png"
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    session = onnxruntime.InferenceSession("/home/rex/Desktop/cv_demo/cv_model/hrnet/hrnet.onnx", providers=["CPUExecutionProvider"])
    pred = session.run(["predict"], {"image": img_tensor.numpy()})[0]
    print(pred)

