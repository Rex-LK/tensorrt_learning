from json import tool
import requests
import matplotlib.pyplot as plt


from tools import * 
from model import *
if __name__ == "__main__":

    detr = DETRdemo(num_classes=91)
    state_dict = torch.load('detr_demo.pth') 
    detr.load_state_dict(state_dict)
    detr.eval()

    im = Image.open("demo.jpg")
    
    #记录原图大小,方便进行后处理
    img = transform(im).unsqueeze(0)
    pred = detr(img)
    # 尽可能的将后后处理放在onnx
    print("---------")
    scores = pred[0][:,0:91]
    print(scores.shape)
    # 需要恢复成原图大小
    bboxes =  pred[0][:,91:]
    print(bboxes.shape)
    keep = scores.max(-1).values > 0.7
    print(keep)
    scores = scores[keep]
    bboxes = bboxes[keep]
    print(bboxes)
    fin_bboxes = rescale_bboxes(bboxes, im.size)
    plot_results(im, scores, fin_bboxes)
