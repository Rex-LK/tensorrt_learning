import paddle
import paddle.nn as nn

paddle.set_device('cpu')
from PIL import Image


def windows_partition(x, window_size):
    """
    :param x: Tensor,shape = [b,h,w,c]
    :param window_size:int,window size
    :return: x :tensor shape = [num_windows*b,window_size,window_size,c]
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x


# TODO :generate attn mask

def generate_mask(window_size=4, shift_size=2, input_resolution=(8, 8)):
    H, W = input_resolution
    img_mask = paddle.zeros([1, H, W, 1])

    h_slice = [slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)]  # a[slice(..)] = a[0:-window_size]

    w_slice = [slice(0, -window_size),
               slice(-window_size, -shift_size),
               slice(-shift_size, None)]  # a[slice(..)] = a[0:-window_size]

    cnt = 0
    for h in h_slice:
        for w in w_slice:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    window_mask = windows_partition(img_mask, window_size=window_size)
    window_mask = window_mask.reshape([-1, window_size * window_size])

    attn_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)
    # [n,1,ws*ws] - [n,ws*ws,1]

    attn_mask = paddle.where(attn_mask != 0,
                             paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
    return attn_mask


def main():
    mask = generate_mask()
    print(mask.shape)
    mask = mask.cpu().numpy().astype('uint8')
    for i in range(4):
        for j in range(16):
            for k in range(16):
                print(mask[i, j, k], end='\t')
            print()

        im = Image.fromarray(mask[i, :, :])
        im.save(f'{i}.png')
        print()
        print()
    print()


if __name__ == "__main__":
    main()
