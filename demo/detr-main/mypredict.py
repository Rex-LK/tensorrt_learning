from random import expovariate
import torch.nn as nn
from tools import _make_detr
from tools import *
from PIL import Image
import cv2
from onnxsim import simplify
import onnx
import torch.nn as nn



if __name__ == "__main__":
# detr = MyDert(model_pth,num_classes,device)

    detr = detr_resnet50()
    im = Image.open("/home/rex/Desktop/tensorrt_learning/demo/detr-main/demo.jpg")
    img = transform(im).unsqueeze(0)
    #设置  dilation=True
    pred = detr(img)
    print(pred)
    scores = pred[0][:,0:91]
    # 需要恢复成原图大小
    bboxes =  pred[0][:,91:]
    keep = scores.max(-1).values > 0.7
    scores = scores[keep]
    bboxes = bboxes[keep]

    fin_bboxes = rescale_bboxes(bboxes, im.size)
    plot_results(im, scores, fin_bboxes)


    export_onnx = 1
    onnx_sim = 1
    if export_onnx:
        torch.onnx.export(
        detr,(img,),
        "detr.onnx",
        input_names=["image"],
        output_names=["predict"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes={"image":{0:"batch"},} 
    )

    if onnx_sim:
        input_path="detr.onnx"
        output_path="detr_sim.onnx"
        onnx_model = onnx.load(input_path)
        model_simp, check = simplify(onnx_model,input_shapes={'image': [1, 3, 800, 1066]})
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_path)
        print('finished exporting onnx')








# scores = pred_logits.softmax(-1)[0, :, :-1][0]
# keep = pred_logits.max(-1).values > 0.7
# scores = pred_logits[keep]
# bboxes = bboxes[keep]
# fin_bboxes = rescale_bboxes(bboxes, im.size)
# plot_results(im, scores, fin_bboxes)


# pred = pred.unsqueeze(0)
# print(pred.shape)
# torch.onnx.export(
#     detr, (img,),
#     "detr.onnx",
#     input_names=["image"],
#     output_names=["predict"],
#     opset_version=12,
#     dynamic_axes={"image": {0: "batch"}, "predict": {0: "batch"}, }
# )
