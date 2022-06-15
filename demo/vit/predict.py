import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = 'cpu'
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_path = "./demo.jpg"
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = create_model(num_classes=5, has_logits=False).to(device)
    model_weight_path = "vit-model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        # output = torch.squeeze(model(img.to(device))).cpu()
        output = model(img.to(device))
        print(output)
        # predict = torch.softmax(output, dim=0)

if __name__ == '__main__':
    main()
