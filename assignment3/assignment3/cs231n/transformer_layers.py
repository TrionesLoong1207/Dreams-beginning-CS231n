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
    编码序列中标记位置的信息。 在此情况下, 该层没有可学习的参数, 因为它是正弦和正弦的简单函数。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence


        中文对照：
        构造位置编码层。

        输入: - embed_dim: 嵌入维的大小 - Dropout（随机失活）: Dropout（随机失活） 值 - max_len: 新进序列的最大可能长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        # [中文] 创建一个“ 批量尺寸” 1 (which will broadcast across all examples in the batch) 的数组。
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # [中文] TODO: 构造所描述的位置编码数组Transformer_Captioning.ipynb。目标是每行交替使用正弦和余弦,并配有0,  0,  2,
        # 2,  4,  4,等等,直到embed_dim。这个精确的规格当然有点武断,但这是自动升级器所期望的。参考一下,我们的解决方案低于5代码行。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        # [中文] 确保位置编码以模型参数(mostly for completeness) 保存。
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
        元素向输入序列添加位置嵌入。

        输入: - x: 输入到位置编码器模型的序列, 形状 (N, S, D), N 是批量大小, S 是序列长度, D 正在嵌入 dimm 返回: - 输出: 输入序列+位置编码, 形状为 (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建一个占位符, 由下面的代码覆盖。
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # [中文] TODO: 将索引输入位置编码, 并将相应的编码添加到输入序列中。 以后不要忘了应用 Dropout（随机失活）。 这只需要几行代码。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
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
    执行“注意是所有你需要的” 所引入的简化蒙蔽注意的模型层 (https://arxiv.org/abs/1706.03762).)

    用法: attn = 多头注意 ( embed_dim, num_heads = 2 )

    # 自控数据 = torch.randn(batch_size, sequence_length, embed_dim) self_attn_output = attn(克里=数据,键=数据,值=数据)

    # 注意,使用两个输入 other_data = torch.randn(batch_size, sequence_length, embed_dim) attn_output = attn(克里=数据,键= other_data,值= other_data )
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability


        中文对照：
        构造一个新的多头注意层。

        输入: - embed_dim: 符号嵌入的尺寸 - num_heads: 注意头数 - Dropout（随机失活）: Dropout（随机失活） 概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        # [中文] 我们将为您初始化这些层, 因为交换命令会影响随机数字生成 (and therefore your exact outputs relative to the
        # autograder)。 注意这些层使用偏差术语, 但这不是绝对必要的 (and varies by implementation)。
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
        计算所提供数据的掩码关注输出, 平行计算所有关注头。

        在下面的形状定义中,N是批量大小,S是源序列长度,T是目标序列长度,E是嵌入尺寸。

        输入: - 查询: 输入数据用作查询, 形状(N, S, E) - 键: 输入的数据用作键, 形状 (N, T, E) - 值: 形状 (N, T, E) - attn_mask 的输入数据用作数值: 元件 (S, T) 的数组, 其中 mask[i,j] = 0 表示源代码中的符号i 不应影响目标中的符号 j。

        返回: - 输出: 形状 (N, S, E) 的 Tensor 表示根据使用键和查询计算到的注意权重,将数据按数值加权组合。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建一个占位符, 由下面的代码覆盖。
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
        # [中文] TODO: 使用给定方程式执行多头关注Transformer_Captioning.ipynb。一些提示:1你会想分裂 你的形状从(N, T,
        # E)插入(N、T、H、E/H),其中H为头数。2函数torch.matmul允许您进行分批的矩阵乘法。 例如, 您可以通过 (N、 H、 E/ H、 T)
        # 生成形状(N, H, T,
        # T)。更多例子见https://pytorch.org/docs/stable/generated/torch.matmul.html
        # 3申请attn_mask,考虑如何修改评分,以防止某个值影响输出。具体来说,PyTorrch 函数masked_fill也许会派上用场

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
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
        简单的双层种子前向网络,有 Dropout（随机失活） 和 ReLU 激活。

        输入: - embed_dim: 输入和输出嵌入的尺寸 - ffn_dim: feedforward 网络中的隐藏尺寸 - Dropout（随机失活）: Dropout（随机失活） 概率
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
         feedforward 网络 前向传播。

        输入: - x: 形状 (N, T, D) 的输入强度

        返回: - 退出: 输出数组与输入的形状相同
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
    一个转换器解码器的单层, 用于转换器解碼器。
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
        构造一个变形DecoderLayer 实例。

        输入: - input_dim: 输入中预期的特性数。 - num_heads:注意头数 - dim_feedforward: feedforward 网络模型的尺寸。 -Dropout（随机失活）: Dropout（随机失活） 值。
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
        通过解码层传递输入(and mask)。

        输入: - tgt: 解码器层的序列, 形状 (N, T, D) - 内存: 编码器末层的序列 形状 (N, S, D) - tgt_mask: 目标序列中隐藏的部件, 形状(T, T)

        返回: - 退出: 形状 (N, T, W) 的变形器特性
        """

        # Self-attention block (reference implementation)
        # [中文] 自控区块 (reference implementation)
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
        # [中文] TODO: 通过执行其余两个子层完成解码层:(1)以编码器输出为内存的交叉注意区块,以及(2)每个区块应采用与以上所实施的自省相同的结构。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

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
    将图像分割成图像块, 并将每个图像块投射到嵌入矢量。 用作 Vision 变换器 (ViT) 的输入层。

    输入: - img_size: 代表输入图像的高度/宽度的整数 (assumes square image) - patch_size: 代表每个图像块 (square patch) 的高度/ 宽度的全数。 - in_channels:输入图像通道的数目( e.g ., 3 为 RGB) - embed_dim:线性嵌入空间的尺寸。
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
        # [中文] 向嵌入尺寸投放平坦图像块的线性投影
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        Forward pass for patch embedding.

        Inputs:
        - x: Input image tensor of shape (N, C, H, W)

        Returns:
        - out: Patch embeddings with shape (N, num_patches, embed_dim)


        中文对照：
         图像块嵌入时 前向传播。

        输入数: - x: 输入图像的形状 (N, C, H, W)

        返回: - 退出: 形状 (N, num_patches, embed_dim) 的图像块嵌入
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
        # [中文] TODO: 将图像分隔为形状 (C x patch_size x patch_size) 的非重叠图像块, 并将其重新排列为形状 (N,
        # num_patches, patch_dim)。 不要使用环形。 相反, 您可能会发现 torch.reshape 和 torch.permute
        # 有助于此步骤。 一旦图像块被平整, 就会使用投影层将其嵌入潜伏矢量。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        return out




class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.


    中文对照：
    一个转换器编码器的单层, 与变换器 Encoder 一起使用。
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
        构造一个变形器 EncoderLayer 实例。

        输入: - input_dim: 输入中预期的特性数。 - num_heads:注意头数。 -- dim_feedforward: feedforward 网络模型的尺寸。 -Dropout（随机失活）: Dropout（随机失活） 值。
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
        通过编码器层传递输入(and mask)。

        输入: - src: 编码器层的序列, 形状 (N, S, D) - src_mask: 源序列的部件, 形状(S, S)

        返回: - 退出: 形状 (N, S, D) 的变形器特性
        """
        ############################################################################
        # TODO: Implement the encoder layer by applying self-attention followed    #
        # by a feedforward block. This code will be very similar to decoder layer. #
        ############################################################################
        # [中文] TODO: 通过应用自我注意和进料前方块来执行编码器层。 此代码将与解码器层非常相似。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        return src
