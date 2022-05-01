# CV学习笔记之Vision Transformer

## 1、前言

### 		Transformer是目前深度学习比较热门的技术之一，在学习过程中发现，网络的原理可以很好的理解，但苦于调包，没由亲手实现过，巧合之下刚好在paddle和B站下看到了有关的视频，视频基本上是从0开始讲解原理和代码，一步一步的实现vit,于是便将其中的代码与笔记记录下来。

### [pallle基础transformer课](https://aistudio.baidu.com/aistudio/education/group/info/25102)

### [B站视频讲解](https://www.bilibili.com/video/BV1Jh411Y7WQ/?spm_id_from=333.788 )

## 2、学习内容

### 	v	it模型的在paddle的课程中已经讲的很明白了，此外，也可以从其他视频中学习，这里推荐

### <u>霹雳吧啦[Wz](https://www.bilibili.com/video/BV1Jh411Y7WQ/?spm_id_from=333.788 )</u>在小破站称之为导师的up主，由兴趣的小伙伴们可以看看。

## 2.1、vit模型

## 		vit的基本结构如下图所示，本次记录的代码主要是右边的EmbededPathches、MultiHead Attention 这个两个部分。

<img src="/home/rex/Pictures/Screenshot from 2022-03-27 19-59-16.png" alt="Screenshot from 2022-03-27 19-59-16" style="zoom:67%;" />





### 			更为详细的图片可以参考霹雳吧啦Wz绘制的图片，如下如所示:

![Screenshot from 2022-03-27 20-06-30](/home/rex/Pictures/Screenshot from 2022-03-27 20-06-30.png)

​	

## 2.2、vit代码记录

### 			如果对vit还不是很了解的同学，可以多刷几个讲解vit模型的视频，对vit模型有了一定的了解之后，便可以参照以下代码，自动实现一个vit模型。

## 2.2.1 、vit类

### 			首构建vit类，这个类包含了整个vit模型的基本流程，而且可以看出，padlle 的风格与torch是一致的。

```python
import paddle
import paddle.nn as nn

class vit(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,):
        super(ViT, self).__init__()
        # 将输入进来的图片转化为一个一个的patch
        self.patch_embedding = PatchEmbedding(image_size, 
                                              patch_size, 
                                              in_channels, 
                                              embed_dim, 
                                              dropout)

        # 初始化多头注意力机制
        self.encoder = Encoder( embed_dim,
                                num_heads, 
                                qkv_bias,
                                mlp_ratio,
                                dropout, 
                                attention_dropout,
                                depth )

        # 最后进行分类
        self.classifier = Classify(embed_dim, dropout, num_classes)

    def forward(self, x):
        # input [N, C, H', W']
        x = self.patch_embedding(x) #[N, C * H + 1, embed_dim]
        x = self.encoder(x)         #[N, C * H + 1, embed_dim]
        x = self.classifier(x[:, 0, :])      #[N, num_classes]
        return x

```

## 2.2.1 、**PatchEmbedding**类

```python
class PatchEmbedding(nn.Layer):
    def __init__(self,
                image_size = 224,
                patch_size = 16,
                in_channels = 3,
                embed_dim = 768,
                dropout = 0.):
        super(PatchEmbedding, self).__init__()
		
        # 14 * 14 
        n_patches = (image_size // patch_size) * (image_size // patch_size)
		
        # 用一个卷积提取patches 
        self.patch_embedding = nn.Conv2D(in_channels = in_channels,
                                         out_channels = embed_dim,
                                         kernel_size = patch_size,
                                         stride = patch_size)
        
        self.dropout=nn.Dropout(dropout)

        #add class token
        self.cls_token = paddle.create_parameter(
                                        shape = [1, 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.Constant(0)
                                        #常量初始化参数，value=0， shape=[1, 1, 768]
                                        )

        #add position embedding
        self.position_embeddings = paddle.create_parameter(
                                        shape = [1, n_patches + 1, embed_dim],
                                        dtype = 'float32',
                                        default_initializer = paddle.nn.initializer.TruncatedNormal(std = 0.02)
                                        #随机截断正态（高斯）分布初始化函数
                                        )

    def forward(self, x):
        x = self.patch_embedding(x) #[N, C, H', W',]  to  [N, embed_dim, H, W]卷积层
        x = x.flatten(2)            #[N, embed_dim, H * W]
        x = x.transpose([0, 2, 1])  #[N, H * W, embed_dim]

        cls_token = self.cls_token.expand((x.shape[0], -1, -1)) #[N, 1, embed_dim]
        x = paddle.concat((cls_token, x), axis = 1)             #[N, H * W + 1, embed_dim]
        x = x + self.position_embeddings                        #[N, H * W + 1, embed_dim]
        x = self.dropout(x)

        return x

```

## 2.2.2  Encoder类

### 		通过多个堆叠注意力模块，可以得到vit的encode部分

```python
class Encoder(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads, 
                 qkv_bias,
                 mlp_ratio,
                 dropout, 
                 attention_dropout,
                 depth):
        super(Encoder, self).__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                        num_heads, 
                                        qkv_bias,
                                        mlp_ratio,
                                        dropout, 
                                        attention_dropout)
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)# or nn.Sequential(*layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):n_patches
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class EncoderLayer(nn.Layer):
    def __init__(self, 
                 embed_dim,
                 num_heads, 
                 qkv_bias,
                 mlp_ratio,
                 dropout, 
                 attention_dropout
                 ):
        super(EncoderLayer, self).__init__()
        #Multi Head Attention & LayerNorm
        w_attr_1, b_attr_1 = self._init_weights()
        self.attn_norm = nn.LayerNorm(embed_dim, 
                                      weight_attr = w_attr_1,
                                      bias_attr = b_attr_1,
                                      epsilon = 1e-6)
        self.attn = Attention(embed_dim,
                              num_heads,
                              qkv_bias,
                              dropout,
                              attention_dropout)

        #MLP & LayerNorm
        w_attr_2, b_attr_2 = self._init_weights()
        self.mlp_norm = nn.LayerNorm(embed_dim,
                                     weight_attr = w_attr_2,
                                     bias_attr = b_attr_2,
                                     epsilon = 1e-6)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x                   #[N, H * W + 1, embed_dim]
        x = self.attn_norm(x)   #Attention LayerNorm
        x = self.attn(x)        #[N, H * W + 1, embed_dim]
        x = h + x               #Add

        h = x                   #[N, H * W + 1, embed_dim]
        x = self.mlp_norm(x)    #MLP LayerNorm
        x = self.mlp(x)         #[N, H * W + 1, embed_dim]
        x = h + x               #[Add]
        return x

```

## 2.2.3  Attention类

```python
class Attention(nn.Layer):
    def __init__(self,
                 embed_dim, 
                 num_heads, 
                 qkv_bias, 
                 dropout, 
                 attention_dropout):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(embed_dim / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads
        self.scales = self.attn_head_size ** -0.5

        #计算qkv矩阵
        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim, 
                             self.all_head_size * 3, # weight for Q K V
                             weight_attr = w_attr_1,
                             bias_attr = b_attr_1 if qkv_bias else False)

        #mlp
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(embed_dim,
                              embed_dim, 
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.ini次记录的代码主要是右边的tializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        #input size  [N, ~, embed_dim]
        new_shape = x.shape[0:2] + [self.num_heads, self.attn_head_size]
        #reshape size[N, ~, head, head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        #transpose   [N, head, ~, head_size]
        return x

    def forward(self, x):
        #input x = [N, H * W + 1, embed_dim]
        qkv = self.qkv(x).chunk(3, axis = -1)           #[N, ~, embed_dim * 3]  list
        q, k, v = map(self.transpose_multihead, qkv)    #[N, head, ~, head_size]
        
        attn = paddle.matmul(q, k, transpose_y = True)  #[N, head, ~, ~]
        attn = self.softmax(attn * self.scales)         #softmax(Q*K/(dk^0.5))
        attn = self.attn_dropout(attn)                  #[N, head, ~, ~]
        
        z = paddle.matmul(attn, v)                      #[N, head, ~, head_size]
        z = z.transpose([0, 2, 1, 3])                   #[N, ~, head, head_size]
        new_shape = z.shape[0:2] + [self.all_head_size]
        z = z.reshape(new_shape)                        #[N, ~, embed_dim]
        z = self.proj(z)                                #[N, ~, embed_dim]
        z = self.proj_dropout(z)                        #[N, ~, embed_dim]

        return z

```

## 2.2.4  Attention类

```python
class Mlp(nn.Layer):
    def __init__(self,
     bias_attr            embed_dim,
                 mlp_ratio,
                 dropout):
        super(Mlp, self).__init__()
        #fc1
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim, 
                            int(embed_dim * mlp_ratio), 
                            weight_attr = w_attr_1, 
                            bias_attr = b_attr_1)
        #fc2
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                            embed_dim, 
                            weight_attr = w_attr_2, 
                            bias_attr = b_attr_2)

        self.act = nn.GELU()#GELU > ELU > ReLU > sigmod
     bias_attr   self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())  
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)         #[N, ~, embed_dim]
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)         #[N, ~, embed_dim]
        x = self.dropout2(x)
        return x

```

## 2.2.5 分类

```python
class Classify(nn.Layer):
    def __init__(self, embed_dim, dropout, num_classes):
        super(Classify, self).__init__()
        #fc1
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim, 
                            embed_dim,
                            weight_attr = w_attr_1,
                            bias_attr = b_attr_1)
        #fc2
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(embed_dim, 
                            num_classes,
                            weight_attr = w_attr_2,
                            bias_attr = b_attr_2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()  

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

```

# 2.3、run起来

```python
def main():
        ins = paddle.randn([1, 3, 224, 224])
        model = ViT()
        out = model(ins)
        print(out.shape)
        paddle.summary(model, (1, 3, 224, 224))

if __name__ == "__main__":
    main()
```

# 3、总结

### 			刚开始学习transfomer时，总是对特征层的维度变化不清晰，只有自己跑过程序，对里面的变量逐一查看过后，才能对网络的架构有更深层次的了解。