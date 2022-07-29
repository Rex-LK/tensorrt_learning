import cv2
import matplotlib.pyplot as plt
import numpy as np
def inv_mat(M):
    k = M[0,0],
    b1 = M[0,2]
    b2 = M[1,2],
    k = list(k)[0]
    return np.array([[1/k,0,-b1/k],
                    [0,1/k,-b2[0]/k]])

def transform(image,dst_size):
    oh,ow = image.shape[:2]
    dh,dw = dst_size
    scale = min(dw/ow,dh/oh)
    
    M = np.array([
        [scale,0,-scale * ow * 0.5 + dw * 0.5],
        [0,scale,-scale * oh * 0.5 + dh * 0.5]
    ])
    return cv2.warpAffine(image,M,dst_size),M,inv_mat(M)

img = cv2.imread("/home/rex/Desktop/notebook/warpaffine/keji2.jpeg")
img_d,M,inv= transform(img,(640,640))

plt.subplot(1,2,1)
plt.title("WarpAffine")
plt.imshow(img_d[...,::-1])

img_s = cv2.warpAffine(img_d,inv,img.shape[:2][::-1])
print(img_s)
plt.subplot(1,2,2)
plt.title("or")
plt.imshow(img_s[...,::-1])