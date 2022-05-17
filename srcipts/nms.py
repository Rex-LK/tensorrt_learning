from cgitb import reset
from typing import overload
import numpy as np
 
 
boxes=np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]]) 
 
 
def py_cpu_nms(dets, thresh):
 
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
 
 
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
 
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
       
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
 
        idx = np.where(ious<=thresh)[0]
        index = index[idx+1]   # because index start from 1
 
    return keep

def my_nms(det,thresh):
    x1 = det[:,0]
    y1 = det[:,1]
    x2 = det[:,2]
    y2 = det[:,3]
    scores = det[:,4]
    index = scores.argsort()[::-1]
    areas = (x2-x1+1)*(y2-y1+1)
    result = []
    while index.size:
        i = index[0]
        result.append(i)

        x11 = np.maximun(x1[i],x1[index[1:]])
        y11 = np.maximun(y1[i],y1[index[1:]])
        x22 = np.maximun(x2[i],x2[index[1:]])
        y22 = np.maximun(x2[i],x2[index[1:]])

        w = np.maximun(0,x22-x11+1)
        h = np.maximun(0,y22-y11+1)
        
        overlaps = w*h

        ious = overlaps/(areas[i]+areas[index[1:]] - overlaps)

        #选择第一个小于阈值的位置
        idx = np.where(ious<=thresh)[0]

        index = index[idx+1]
    return result


import matplotlib.pyplot as plt
def plot_bbox(dets, c='k'):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    plt.plot([x1,x2], [y1,y1], c)
    plt.plot([x1,x1], [y1,y2], c)
    plt.plot([x1,x2], [y2,y2], c)
    plt.plot([x2,x2], [y1,y2], c)
    plt.title(" nms")
 
    
plt.figure(1)

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
 
plt.sca(ax1)
plot_bbox(boxes,'k')   # before nms
 
keep = py_cpu_nms(boxes, thresh=0.7)
plt.sca(ax2)
plot_bbox(boxes[keep], 'r')# after nms

plt.show()





def numpy_conv(inputs, myfilter):
    h_ori, w_ori = inputs.shape
    h_k, w_k= myfilter.shape
    
    h_new, w_new = h_ori-h_k+1,w_ori-w_k+1
    result = np.zeros((h_new, w_new))
    
    for row in range(0, h_new, 1):
        for col in range(0, w_new, 1):
            # 池化大小的输入区域
            cur_input = inputs[row:row+h_k, col:col+w_k]
            #和核进行乘法计算
            cur_output = cur_input * myfilter
            #再把所有值求和
            conv_sum = np.sum(cur_output)
            #当前点输出值
            result[row, col] = conv_sum
    return result         

#s=1 p=0
def my_conv(inputs, filter):
    in_h,in_w = inputs.shape
    k_h,k_w = filter.shape

    out_h = in_h - k_h + 1
    out_w = in_w - k_w + 1
    result= np.zeros((out_h,out_w))
    for raw in range(out_h):
        for col in range(out_w):
            cur_input = inputs[raw:raw+k_h,col:col+k_w]
            cur_sum = np.sum(cur_input * filter)
            result[raw][col] = cur_sum
    
    return result