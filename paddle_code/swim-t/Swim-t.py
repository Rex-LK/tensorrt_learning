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
        self.attn = WindowAttention(dim, window_size, num_heads)
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
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

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
