import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.


    中文对照：
    为序列中的 token 编码位置信息。这里使用固定的正弦、余弦函数，因此本层没有可学习参数。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence


        中文对照：
        构造位置编码层。embed_dim 是嵌入维度，dropout 是随机失活概率，max_len 是允许的最大输入序列长度。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        # [中文] 为位置编码增加长度为 1 的批维度，使其可通过广播用于批内所有样本。
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # [中文] 待完成：按 Transformer_Captioning.ipynb 定义构造位置编码。每行交替使用正弦与余弦，指数中的序号按 0、0、2、2、4、4 等排列，直到覆盖
        # embed_dim。请严格匹配题目约定，自动测试依赖这一具体形式；参考实现不到五行。
        position = torch.arange(max_len).unsqueeze(1)
        dims = torch.arange(0, embed_dim, 2)
        
        angles = position * (10000 ** (-dims / embed_dim))

        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        # [中文] 将位置编码注册到模型，使其随模型参数一起保存。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)


        中文对照：
        将位置编码逐元素加到输入序列上。x 和输出的形状均为 (N,S,D)，其中 N 为批大小，S 为序列长度，D 为嵌入维度。
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建占位值，后续由你实现的代码覆盖。
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # [中文] 待完成：取出与输入序列位置对应的位置编码，加到输入上，然后应用 Dropout。参考实现只需几行。
        pe = self.pe
        x_pe = pe[:, :S, :] + x
        output = self.dropout(x_pe)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)


    中文对照：
    实现 Attention Is All You Need 中掩码注意力的简化版本。query、key、value 来自同一序列时是自注意力；query 来自一条序列而 key、value 来自另一条序列时，可用于交叉注意力。调用方式和论文链接见英文原说明。
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability


        中文对照：
        构造多头注意力层。embed_dim 是 token 的嵌入维度，num_heads 是注意力头数，dropout 是随机失活概率。
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        # [中文] 层的初始化已提供。调整初始化顺序会影响随机数消耗顺序，进而影响与自动测试参考值的逐项比较。本实现使用偏置，其他实现也可能不使用。
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.


        中文对照：
        并行计算各注意力头并得到带掩码的注意力输出。query 为 (N,S,E)，key、value 为 (N,T,E)，attn_mask 为 (S,T)，掩码中的 0 标记禁止参与注意力的位置；输出为 (N,S,E)，是按 query/key 计算的权重对 value 加权汇总的结果。N 为批大小，S、T 分别是查询与键值序列长度，E 为嵌入维度。掩码行列含义以当前输入的查询、键值顺序为准。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建占位值，后续由你实现的代码覆盖。
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # [中文] 待完成：按 notebook 公式实现多头注意力。原题提示：将 (N,T,E) 拆成 (N,T,H,E/H)，其中 H 为头数；torch.matmul
        # 可完成批量矩阵乘法，例如 (N,H,T,E/H) 与 (N,H,E/H,T) 相乘得到 (N,H,T,T)。应用 attn_mask 时要防止被屏蔽位置影响输出，可参考
        # masked_fill。更多用法见英文文档链接。
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        H = self.n_head

        Q = Q.reshape(N, S, H, self.head_dim).transpose(1, 2)
        K = K.reshape(N, T, H, self.head_dim).transpose(1, 2)
        V = V.reshape(N, T, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attn_drop(attention_weights)
        out = torch.matmul(attention_weights, V)

        out = out.transpose(1, 2).reshape(N, S, E)
        output = self.proj(out)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Simple two-layer feed-forward network with dropout and ReLU activation.

        Inputs:
         - embed_dim: Dimension of input and output embeddings
         - ffn_dim: Hidden dimension in the feedforward network
         - dropout: Dropout probability


        中文对照：
        带 ReLU 和 Dropout 的双层前馈网络。embed_dim 是输入输出嵌入维度，ffn_dim 是隐藏层维度，dropout 是随机失活概率。
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Inputs:
        - x: Input tensor of shape (N, T, D)

        Returns:
        - out: Output tensor of the same shape as input


        中文对照：
        前馈网络的前向传播。输入 x 为 (N,T,D)，输出 out 与输入形状相同。
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.


    中文对照：
    Transformer 解码器中的一个解码器层。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.


        中文对照：
        构造 TransformerDecoderLayer。input_dim 为输入特征维度，num_heads 为注意力头数，dim_feedforward 为前馈网络隐藏维度，dropout 为随机失活概率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, D)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)


        中文对照：
        将输入与掩码送入解码器层。tgt 为 (N,T,D) 的目标序列，memory 为编码器最后一层的输出 (N,S,D)，tgt_mask 为 (T,T) 的目标序列掩码。返回 Transformer 特征；英文原注释将输出形状记为 (N,T,W)。
        """

        # Self-attention block (reference implementation)
        # [中文] 自注意力子层，代码已作为参考实现提供。
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO: Complete the decoder layer by implementing the remaining two       #
        # sublayers: (1) the cross-attention block using the encoder output as     #
        # memory, and (2) the feedforward block. Each block should follow the      #
        # same structure as self-attention implemented just above.                 #
        ############################################################################
        # [中文] 待完成：补全解码器层剩余两个子层：以编码器输出为 memory 的交叉注意力，以及前馈网络。每个子层采用与上面自注意力相同的整体结构。
        shortcut = tgt
        tgt = self.cross_attn(query=tgt, key=memory, value=memory)
        tgt = self.dropout_cross(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_cross(tgt)

        shortcut = tgt
        tgt = self.ffn(tgt)
        tgt= self.dropout_ffn(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_ffn(tgt)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        return tgt


class PatchEmbedding(nn.Module):
    """
    A layer that splits an image into patches and projects each patch to an embedding vector.
    Used as the input layer of a Vision Transformer (ViT).

    Inputs:
    - img_size: Integer representing the height/width of input image (assumes square image).
    - patch_size: Integer representing height/width of each patch (square patch).
    - in_channels: Number of input image channels (e.g., 3 for RGB).
    - embed_dim: Dimension of the linear embedding space.


    中文对照：
    ViT 的图像块嵌入层：将图片划分为图像块，再将每块投影为嵌入向量。img_size、patch_size 分别为正方形图片和图像块的边长，in_channels 为输入通道数，embed_dim 为嵌入维度。
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection of flattened patches to the embedding dimension
        # [中文] 将展平的图像块线性投影到嵌入维度。
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        Forward pass for patch embedding.

        Inputs:
        - x: Input image tensor of shape (N, C, H, W)

        Returns:
        - out: Patch embeddings with shape (N, num_patches, embed_dim)


        中文对照：
        图像块嵌入层的前向传播。输入图像 x 为 (N,C,H,W)，输出图像块嵌入 out 为 (N,num_patches,embed_dim)。
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO: Divide the image into non-overlapping patches of shape             #
        # (C x patch_size x patch_size), and rearrange them into a tensor of       #
        # shape (N, num_patches, patch_dim). Do not use a for-loop.                #
        # Instead, you may find torch.reshape and torch.permute helpful for this   #
        # step. Once the patches are flattened, embed them into latent vectors     #
        # using the projection layer.                                              #
        ############################################################################
        # [中文] 待完成：将图片划分为不重叠的图像块，每块形状为 (C,patch_size,patch_size)，重排成 (N,num_patches,patch_dim)。不使用 for
        # 循环，可参考 torch.reshape 和 torch.permute；展平后用 projection 层映射到嵌入空间。
        patch_size = self.patch_size
        patch_dim = self.patch_dim
        num_patches = self.num_patches

        x_division = x.reshape(
            N, C,
            H // patch_size, patch_size,
            W // patch_size, patch_size
        )

        x_division = x_division.permute(0, 2, 4, 1, 3, 5)
        x_patches = x_division.reshape(N, num_patches, patch_dim)

        out = self.proj(x_patches)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return out




class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.


    中文对照：
    Transformer 编码器中的一个编码器层。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads.
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.


        中文对照：
        构造 TransformerEncoderLayer。input_dim 为输入特征维度，num_heads 为注意力头数，dim_feedforward 为前馈网络隐藏维度，dropout 为随机失活概率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - src: the sequence to the encoder layer, of shape (N, S, D)
        - src_mask: the parts of the source sequence to mask, of shape (S, S)

        Returns:
        - out: the Transformer features, of shape (N, S, D)


        中文对照：
        将 src 及掩码送入编码器层。src 为 (N,S,D)，src_mask 为 (S,S)；返回同形状的 Transformer 特征 out。
        """
        ############################################################################
        # TODO: Implement the encoder layer by applying self-attention followed    #
        # by a feedforward block. This code will be very similar to decoder layer. #
        ############################################################################
        # [中文] 待完成：依次应用自注意力和前馈网络，完成编码器层。整体写法与解码器层相近。
        shortcut = src
        src = self.self_attn(src, src, src, src_mask)
        src = self.dropout_self(src)
        src = src + shortcut
        src = self.norm_self(src)

        shortcut = src
        src = self.ffn(src)
        src = self.dropout_ffn(src)
        src = src + shortcut
        src = self.norm_ffn(src)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return src
