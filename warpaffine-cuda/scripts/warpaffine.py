# p1 ~ p4 的值
p1 = 1
p2 = 2 
p3 = 8
p4 = 12

# p1~p4 位四个相邻的像素，围起来的面积为1

p = 0.6,0.8

p1_w = (1 - p[0]) * (1 - p[1])
p2_w = p[0] * (1 - p[1])
p3_w = (1 - p[0]) * p[1]
p4_w = p[0] * p[1]

res = p1 * p1_w + p2 * p2_w + p3 * p3_w + p4 * p4_w 
print(res)


import cv2
import numpy as np
import matplotlib.pyplot as plt 

def myWarpaffine(img,M,d_size,constant=(0, 0, 0)):
    # 其中 M 为 原图变为目标图的变换矩阵，将缩放后的图恢复到原图需要取逆变换
    M_inv = cv2.invertAffineTransform(M)
    constant = np.array(constant)
    o_h, o_w = img.shape[:2]
    d_h, d_w = d_size
    dst_img = np.full((d_h, d_w, 3), constant, dtype=np.uint8)
    o_range = lambda p: p[0] >= 0 and p[0] < o_w and p[1] >= 0 and p[1] < o_h
    
    for y in range(d_h):
        for x in range(d_w):
            # 缩放后图上的一点
            homogeneous = np.array([[x, y, 1]]).T
            
            # 恢复到原图的尺寸
            ox, oy = M_inv @ homogeneous
            # p1   p2
            #    p
            # p3   p4
            #np.floor(ox) 类型为np.float64
            low_ox = int(np.floor(ox))
            low_oy = int(np.floor(oy))
            high_ox = low_ox + 1
            high_oy = low_oy + 1
            
            p = ox - low_ox, oy - low_oy
            p1_w = (1 - p[0]) * (1 - p[1])
            p2_w = p[0] * (1 - p[1])
            p3_w = (1 - p[0]) * p[1]
            p4_w = p[0] * p[1]
            
            p1 = low_ox, low_oy
            p2 = high_ox, low_oy
            p3 = low_ox, high_oy
            p4 = high_ox, high_oy
            # 避免超出图片范围
            p1_value = img[p1[1], p1[0]] if o_range(p1) else constant
            p2_value = img[p2[1], p2[0]] if o_range(p2) else constant
            p3_value = img[p3[1], p3[0]] if o_range(p3) else constant
            p4_value = img[p4[1], p4[0]] if o_range(p4) else constant
            dst_img[y, x] = p1_w * p1_value + p2_w * p2_value + p3_w * p3_value + p4_w * p4_value    
    return dst_img

if __name__ == "__main__":
    img_o = cv2.imread("/home/rex/Desktop/rex_extra/notebook/warpaffine/keji2.jpeg")
    # 图片旋转中心、旋转角度、缩放倍数
    M = cv2.getRotationMatrix2D((0, 0), -30, 0.6)
    or_test = cv2.warpAffine(img_o, M, (640, 640))
    my_test = myWarpaffine(img_o,M,(640, 640))

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("opencv")
    #BGR -> RGB
    plt.imshow(or_test[...,::-1])

    plt.subplot(1, 2, 2)
    plt.title("pyWarpaffine")
    # BGR -> RGB
    plt.imshow(my_test[...,::-1])