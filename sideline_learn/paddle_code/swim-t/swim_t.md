# CV学习笔记之Swim Transformer

## 1、前言

### 	相对于Vision transform，Swim transformer进一步促进了transformer在视觉领域的应用，本次记录的代码依旧是参考paddle的0基础入门transformer

### [pallle基础transformer课](https://aistudio.baidu.com/aistudio/education/group/info/25102)

## 2、学习内容

### 	Swim Transformer在paddle的课程中已经讲的很细致了，本次记录主要是针对其中的代码实现，同时

### 也推荐<u>霹雳吧啦[Wz](https://www.bilibili.com/video/BV1Jh411Y7WQ/?spm_id_from=333.788 )</u>的视频。

## 2.1 Swim Transformer网络结构

### 	Swim Transformer网络结构如下图所示

<img src="/home/rex/Pictures/Screenshot from 2022-03-27 22-09-43.png" alt="Screenshot from 2022-03-27 22-09-43" style="zoom:67%;" />





### 	其中encode中与vit 不同的是修改了注意力的计算形式，大幅减少了计算量，详细的内容大家可以去看视频，这里主要是复现代码。

## 2.2 Swim Transformer代码记录

## 2.2.1 窗口注意力

```python
import paddle
import paddle.nn as nn

class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [n,embed_dim,h',w']
        x = x.flatten(2)  # [n,embed_dim,h',w']
        x = x.transpose([0, 2, 1])
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])

        # 间隔取特征层
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # 合并
        x = paddle.concat([x0, x1, x2, x3], axis=1)  # [B,h/2,w/2,4*c]

        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 2, 3, 4, 5])
    # [B,h//ws,w//ws,ws,ws,c]
    x = x.reshape([-1, window_size, window_size, C])
    # [B*num_patches,ws,ws,c]
    return x


def window_reverse(window, window_size, H, W):
    B = int(window.shape[0] // (H / window_size * W / window_size))
    x = window.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1]) #[num_window]
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_head = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim,
                             dim * 3)
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_head, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])  # [ B,num_heads,num_patches,dim_head]
        return x

    def forward(self, x):
        B, N, C = x.shape
        # x:[B,num_patches,embed_dim]
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)

        out = paddle.matmul(attn, v)  # [B,num_heads,num_patches,dim_head]
        out = out.transpose([0, 2, 1, 3])
        # out:[B,num_patches,num_head,dim_head ] num_head*dim_head=embed_dim
        out = out.reshape([B, N, C])

        out = self.proj(out)
        return out


class Swinblock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape
        h = x
        x = self.attn_norm(x)

        x = x.reshape([B, H, W, C])
        x_windows = windows_partition(x, self.window_size)
        # [B*num_patches,ws,ws,c]
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = x.reshape([B, H * W, C])
        # [B,H,W,C]

        x = h + x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block = Swinblock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    out = patch_embedding(t)  # [4,56,56,96]
    print("patch_embedding out shape :", out.shape)
    out = swin_block(out)
    print("swin_block out shape :", out.shape)
    out = patch_merging(out)
    print("patch_merging out shape :", out.shape)


if __name__ == "__main__":
    main()

```

### 2.2.2 滑动窗口中的maks

```python
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

```

### 2.2.3 滑动窗口注意力

​	由于作者视频中的代码没有给全，这里跑不起来

```
import paddle
import paddle.nn as nn


class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [n,embed_dim,h',w']
        x = x.flatten(2)  # [n,embed_dim,h',w']
        x = x.transpose([0, 2, 1])
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])

        # 间隔取特征层
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # 合并
        x = paddle.concat([x0, x1, x2, x3], axis=1)  # [B,h/2,w/2,4*c]

        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 2, 3, 4, 5])
    # [B,h//ws,w//ws,ws,ws,c]
    x = x.reshape([-1, window_size, window_size, C])
    # [B*num_patches,ws,ws,c]
    return x


def window_reverse(window, window_size, H, W):
    B = int(window.shape[0] // (H / window_size * W / window_size))
    x = window.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])  # [num_window]
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim,
                             dim * 3)
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])  # [ B,num_heads,num_patches,dim_head]
        return x

    def forward(self, x):
        B, N, C = x.shape
        # x:[B,num_patches,embed_dim]
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)

        relative_position_bias = self.get_relative_position_bias_from_index()
        relative_position_bias = relative_position_bias.reshape(
            [self.window_size * self.window_size, self.window_size * self.window_size, -1])
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask [num_windows,num_patches,num_patches]
            # attn [B*num_pathes,num_heads,num_patches,num_patches]
            attn = attn.reshape([B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])

            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shapep[1]])

        out = paddle.matmui(attn, v)
        out = out.transpose([0, 2, 1, 3])
        out = out.reshpe([B, N, C])
        out = self.proj(out)
        return out


class Swinblock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Window2.2.2 滑动窗口中的maksAttention(dim, window_size, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size)
        else:
            attn_mask = 0

        self.register_buffer('attn_mask', self.attn_mask)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape
        h = x
        x = self.attn_norm(x)
        x = x.reshape([B, H, W, C])

        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window2.2.2 滑动窗口中的maks_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            x = shifted_x


def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block_w_msa = Swinblock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    swin_block_sw_msa = Swinblock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7 // 2)

    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    # out = patch_embedding(t)  # [4,56,56,96]
    # print("patch_embedding out shape :", out.shape)
    # out = swin_block(out)
    # print("swin_block out shape :", out.shape)
    # out = patch_merging(out)
    # print("patch_merging out shape :", out.shape)


if __name__ == "__main__":
    main()

```

## 3、总结

#### 	从transformer到vision transformer再到swim transformer，注意力机制在CV领域得到了飞速的发展，在学习过程中不仅对网络结构有了更加深层次的认识，同时也提高了自己的编程技能，后面后持续关注transformer的进展。
