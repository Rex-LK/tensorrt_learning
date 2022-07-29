import paddle
import paddle.nn as nn



class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_drop=0.):
        super().__init__()
        # self.embed_dim = 96
        self.embed_dim = embed_dim
        # self.num_heads = 4
        self.num_heads = num_heads
        # self.head_dim 每个头输出长度 96/4 = 24
        self.head_dim = int(embed_dim / self.num_heads)
        # 所有头输出的长度
        self.all_head_dim = self.head_dim * self.num_heads
        self.qkv_bias = qkv_bias
        # 输入 96，输出 96*3
        self.qkv = nn.Linear(self.embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if self.qkv_bias is False else None,
                             )
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        # 映射为原token大小
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # x: [B,N,all_head_dim]*3
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B,N,num_heads,head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x: [B,N,num_patches,head_dim]
        return x

    # 将代码与公式配套
    def forward(self, x):
        B, N, _ = x.shape
        # chunk 沿着轴-1将 self.qkv(x) 分成三块
        qkv = self.qkv(x).chunk(3, -1)
        # [B,N,all_head_dim]*3
        q, k, v = map(self.transpose_multi_head, qkv)

        # q,k,v: [B,N,num_patches,head_dim]
        # q*k^t
        attn = paddle.matmul(q, k, transpose_y=True)
        # 除以根号d
        attn = self.scale * attn
        # softmax
        attn = self.softmax(attn)
        attn_weight = attn
        # dropout
        # attn: [B,num_heads,num_patches,num_patches]
        out = paddle.matmul(attn, v)  # softmax(scale*(q*k')))*v
        out = out.transpose([0, 2, 1, 3])
        # attn: [B,num_patches,num_heads,num_dim]
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        # dropout
        return out, attn_weight


def main():
    # 输入数据
    t = paddle.randn([8, 16, 96])
    # 编码维度96
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    # print(model)
    out, w = model(t)
    print(out.shape)
    print(w.shape)


if __name__ == "__main__":
    main()
