import cv2
import numpy as np

def inv_mat(M):
    k = M[0, 0],
    b1 = M[0, 2]
    b2 = M[1, 2],
    return np.array([[1 / k[0], 0, -b1 / k[0]],
                     [0, 1 / k[0], -b2[0] / k[0]]])


def image_transfer(image, dst_size):
    oh, ow = image.shape[:2]
    dh, dw = dst_size
    scale = min(dw / ow, dh / oh)

    M = np.array([
        [scale, 0, -scale * ow * 0.5 + dw * 0.5],
        [0, scale, -scale * oh * 0.5 + dh * 0.5]
    ])
    return cv2.warpAffine(image, M, dst_size), M, inv_mat(M)


if __name__ == '__main__':
    img = cv2.imread("/hrnet/images/person2.png")
    cv2.imshow("img_o", img)
    print(img.shape)
    img_d, M, inv = image_transfer(img, (192, 256))
    print(img_d.shape)
    cv2.imshow("img_d",img_d)
    img_s = cv2.warpAffine(img_d, inv, img.shape[:2][::-1])
    cv2.imshow("img_s",img_s)
    cv2.waitKey(0)