import math
import torch
import torch.nn as nn
from torchvision.transforms.functional import crop
from torch.nn import LayerNorm, Linear, Softmax, ModuleList, GELU, Sequential, Parameter, Dropout

class SelfAttention(nn.Module):

    def __init__(self, dim_input, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = Linear(dim_input, embedding_dim)
        self.key = Linear(dim_input, embedding_dim)
        self.value = Linear(dim_input, embedding_dim)
        self.softmax = Softmax(dim=2)

    def forward(self, inputs):
        # inputs: [batch_size, N, dim_model]
        q = self.query(inputs) # q: [N, embedding_dim]
        k = self.key(inputs) # k: [N, embedding_dim]
        v = self.value(inputs) # v: [N, embedding_dim]

        scores = torch.matmul(q, torch.transpose(k, -2, -1)) / math.sqrt(k.size(-1))
                                                                # scores: [N, N]
        attention = torch.matmul(self.softmax(scores), v)

        return attention # attention: [N, embedding_dim]


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_model, num_head=9):
        super(MultiHeadAttention, self).__init__()
        assert dim_model // num_head != 0, "dim_model必须要能整除num_head数目"
        dim_per_head = dim_model // num_head
        self.num_head = num_head
        self.atten_heads = ModuleList([
            SelfAttention(dim_model, dim_per_head) for _ in range(num_head)
        ])
        self.projection = Linear(num_head * dim_per_head, dim_model)
                    #避免维度不同，同时也添加了非线性层

    def forward(self, inputs):
        # inputs: [batch_size, N, dim_model]
        x = [attn(inputs) for i, attn in enumerate(self.atten_heads)] # x[i]: [batch_size, N, dim_model // num_head]
        x = torch.cat(x, len(x[0].size())-1) # x[i]: [batch_size, N, dim_model ]
        return self.projection(x) # return: [batch_size, N, dim_model ]

class EncoderBlock(nn.Module):

    def __init__(self, dim_model, mlp_dim, num_head=8, dropout=.5):
        super(EncoderBlock, self).__init__()
        assert dim_model // num_head != 0, "dim_model必须要能整除num_head数目"
        self.norm1 = LayerNorm(dim_model)
        self.msa = MultiHeadAttention(dim_model, num_head)
        self.mlp = Sequential(
            Linear(dim_model, mlp_dim),
            Dropout(dropout),
            GELU(),
            Linear(mlp_dim, dim_model),
            Dropout(dropout)
        )
        self.norm2 = LayerNorm(dim_model)

    def forward(self, inputs):
        # inputs: [batch_size, N, dim_model]
        output = self.msa(self.norm1(inputs))
        res = inputs + output

        output = self.norm2(res)
        result = self.mlp(output)
        output = self.mlp(result)
        return res + output # return: [batch_size, N, dim_model ]

class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 batch_size,
                 dim_model,
                 num_layer,
                 num_class,
                 num_head=8,
                 mlp_dim=128,
                 channel=3,
                 dropout=0.5):
        super(VisionTransformer, self).__init__()
        assert image_size // patch_size != 0, "注意这里图片不能完整切分"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch = (image_size // patch_size) ** 2
        dim_patch = patch_size ** 2 *channel
        self.projection = Linear(dim_patch, dim_model)
        self.positionEmbedding = Parameter(torch.Tensor(self.num_patch + 1, dim_model))
        self.classEmbedding = Parameter(torch.Tensor(batch_size, 1, dim_model))
        self.encoder_layers = ModuleList([
            EncoderBlock(dim_model, mlp_dim, num_head, dropout=dropout) for _ in range(num_layer)
        ])
        self.mlpHead = Sequential(
            LayerNorm(dim_model),
            Linear(dim_model, mlp_dim),
            Dropout(dropout),
            GELU(),
            Linear(mlp_dim, num_class),
            Dropout(dropout)
        )

    def extract_patches(self, images):
        # images: [batch_size, channel, input_size, input_size]
        patches = []
        for i in range(self.image_size // self.patch_size):
            for j in range(self.image_size // self.patch_size):
                patches.append(torch.flatten(crop(images, i * self.patch_size, i * self.patch_size, self.patch_size, self.patch_size), 1))
        return torch.stack(patches, 1)
                                                # return: [batch_size, num_patch, patch_size * patch_size * channel]
        # 这里是暴力方案：
        # 这里是先从batch取出每个image，然后切patch，把每一个patch展平为patch_size * patch_size * channel
        # 每张图的patch堆叠起来成一个 num_patch * (patch_size * patch_size * channel)的二维图，
        # 最后按照batch堆叠起来……
        # 本来pytorch好像没有提供任意batch的切分方案，其实这里可以思考按一个batch切分
        # return torch.stack([
        #     torch.stack([
        #         torch.flatten(
        #             image[:, i * patch_size: i * patch_size + patch_size, i * patch_size: i * patch_size + patch_size]
        #         )
        #         for i in self.num_patch], 0)
        #     for image in images], 0)



    def forward(self, inputs):
        # inputs: [batch_size, channel, image_size, image_size]
        patched_inputs = self.extract_patches(inputs) # patched_inputs: [batch_size, num_patch, patch_size * patch_size * channel]
        projected_inputs = self.projection(patched_inputs) # embed_inputs : [batch_size, num_patch, dim_model]

        # embedding part
        embeded_patches = torch.cat([self.classEmbedding, projected_inputs], 1) + self.positionEmbedding # embeded_patches : [batch_size, num_patch + 1, dim_model]
        outputs = None
        for _, layer in enumerate(self.encoder_layers):
            outputs = layer(embeded_patches) # outputs : [batch_size, num_patch + 1, dim_model]
        return self.mlpHead(outputs[:, 0]) # return: [batch_size, num_classes]