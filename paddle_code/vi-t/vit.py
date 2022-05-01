import copy

import charset_normalizer.utils
import paddle
import paddle.nn as nn
from attention import *
paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)


class PatchEmbedding(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.dropout = nn.Dropout(dropout)

        # add_class token
        self.class_token = paddle.create_parameter(shape=[1, 1, embed_dim],
                                                   dtype='float32',
                                                   default_initializer=nn.initializer.Constant(0, ))

        # add position embedding
        self.position_embedding = paddle.create_parameter(shape=[1, n_patches + 1, embed_dim],
                                                          dtype='float32',
                                                          default_initializer=nn.initializer.TruncatedNormal(std=.02))

    def forward(self, x):
        # x: [n,c,h,w]

        cls_tokens = self.class_token.expand([x.shape[0], 1, self.embed_dim])
        x = self.patch_embedding(x)  # [n,embed_dim,h',w']
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat([cls_tokens, x], axis=1)

        x = x + self.position_embedding
        return x


# [N,C,H,W]


class EncoderLayer():
    def __init__(self, embed_dim=768, num_heads=4, qkv_bias=True, mlp_ratio=4.0, dropout=0. ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)

    def forward(self, x):
        h = x  # residual
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class Encoder(nn.Layer):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer()
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class VisualTransformer(nn.Layer):
    def __init__(self, image_size,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_drop=0.,
                 droppath=0.):
        super().__init__()
        self.path_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.encoder = Encoder(embed_dim, depth)
        self.classifer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x:[N,C,H,W]
        N, C, H, W = x.shape
        x = self.path_embedding(x)  # [N,embed_dim,h',w']
        x = x.flatten(2)  # [N,embed_dim,h'*w'] h'*w'=num_patches
        x = x.transpose([0, 2, 1])  # [N,num_patches,embed_dim]
        x = self.encoder(x)
        print(x)
        x = self.classifer(x[:, 0])
        return x


def main():
    vit = VisualTransformer(224)
    print(vit)
    paddle.summary(vit, (4, 3, 224, 224))


if __name__ == "__main__":
    main()
