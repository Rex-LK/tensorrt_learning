"""
PyTorch style implementation
"""
import numpy as np
import torch
import torch.nn.functional as F


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


if __name__ == '__main__':
    # numpy computation
    a = np.arange(1 * 8 * 4 * 4).reshape([1, 8, 4, 4])
    b = pixel_shuffle(a, scale_factor=2).astype(np.int32)
    c = pixel_shuffle_inv(b, scale_factor=2).astype(np.int32)

    # torch computation
    a_torch = torch.arange(2 * 20 * 7 * 7).reshape([2, 20, 7, 7])
    b_torch = F.pixel_shuffle(a_torch, upscale_factor=2)

    a_torch = a_torch.numpy().astype(np.int32)
    b_torch = b_torch.numpy().astype(np.int32)

    # check
    print(np.all(b == b_torch))
    print(np.all(c == a_torch))


