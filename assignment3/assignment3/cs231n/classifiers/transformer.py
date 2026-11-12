import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.


    中文对照：
    描述转换器使用变换器解码器从图像特性中生成字幕。

    变形器接收的输入矢量为 D 大小, vocab 大小为 V, 运行长度 T 的序列, 使用维度 W 的单词矢量, 并运行大小为 N 的小型信箱。
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.


        中文对照：
        构造一个新的描述 Transex 实例。

        投入: -word_to_idx:提供词汇的字典。它包含 V 项。并且将每个字符串绘制成区域中唯一的整数[0, V). -  input_dim:输入图像特征向量的尺寸 D。 - wordvec_dim:字矢量的维度 W。 - num_heads:关注的负责人人数。num_layers:Transformer层的数目。 -max_length:最大可能的序列长度。
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.


        中文对照：
        初始化网络的权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)


        中文对照：
        根据图像特性和描述标记, 返回每个时间步骤可能显示的标记的分布。 请注意, 由于整个描述序列都是同时提供的, 我们隐藏未来的时间步骤。

        输入: - 特征: 图像特征, 形状 (N, D) - 描述: 地面真相说明, 形状(N, T)

        返回: - 评分: 形状(N, T, V) 的每个时间步骤的每个标记的评分
        """
        N, T = captions.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建一个占位符, 由下面的代码覆盖。
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: Implement the forward function for CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have to embed your caption and add positional              #
        #     encoding. You then have to project the image features into the same  #
        #     dimensions.                                                          #
        #  2) You have to prepare a mask (tgt_mask) for masking out the future     #
        #     timesteps in captions. torch.tril() function might help in preparing #
        #     this mask.                                                           #
        #  3) Finally, apply the decoder features on the text & image embeddings   #
        #     along with the tgt_mask. Project the output to scores per token      #
        ############################################################################
        # [中文] TODO: 执行描述转换后的前方函数。 几个提示:1) 您首先必须嵌入您的描述并添加位置编码。 然后您必须将图像特性投影到相同的维度中
        #。2(你必须准备一个面具)(tgt_mask)以掩盖描述中的未来时间步骤。torch.tril()函数可能有助于准备此掩码。 NAME OF
        # TRANSLATORS3最后,在文本和图像嵌入中应用解码器特性tgt_mask。将输出投进到每个标记的分数

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        return scores

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)


        中文对照：
        鉴于图像特征,使用贪婪的解码来预测图像描述。

        输入: - 特征: 图像特征, 形状 (N, D) - max_length: 最大可能的描述长度

        返回: - 描述: 形状 (N, max_length) 的每个示例的描述
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            # [中文] 创建空描述“抗拉” (where all tokens are NULL)。
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            # [中文] 创建部分描述, 仅使用起始标记。
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                # [中文] 预测下一个符号 (ignoring all other time steps)。
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                # [中文] 从词汇中选择最可能的单词 ID。
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                # [中文] 更新我们的总描述和我们目前的部分描述。
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output



class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.


    中文对照：
    愿景转换器 (ViT) 执行。
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        Inputs:
         - img_size: Size of input image (assumed square).
         - patch_size: Size of each patch (assumed square).
         - in_channels: Number of image channels.
         - embed_dim: Embedding dimension for each patch.
         - num_layers: Number of Transformer encoder layers.
         - num_heads: Number of attention heads.
         - dim_feedforward: Hidden size of feedforward network.
         - num_classes: Number of classification labels.
         - dropout: Dropout probability.


        中文对照：
        投入: -img_size: 输入图像的大小(assumed square) . -  patch_size: 每个补丁的大小(assumed square) . -  in_channels:图像频道的数量。 - embed_dim每个补丁的嵌入维度。 - num_layers: 变换编码器层的数目。 - num_heads:关注的负责人人数。dim_feedforward: 向导网络的隐藏大小。 - num_classes: 分类标签数量。 - Dropout（随机失活）:  Dropout（随机失活）概率。
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer to predict class scores from pooled token.
        # [中文] 最终分类层 以预测类分数 从池化后的 token。
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.


        中文对照：
        初始化网络的权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Inputs:
         - x: Input image tensor of shape (N, C, H, W)

        Returns:
         - logits: Output classification logits of shape (N, num_classes)


        中文对照：
         愿景转换器 前向传播。

        输入数: - x: 输入图像的形状 (N, C, H, W)

        返回: - 登录: 形状 (N, num_classes) 的输出分类登录
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # TODO: Implement the forward pass of the Vision Transformer.             #
        # 1. Convert the input image into a sequence of patch vectors.            #
        # 2. Add positional encodings to retain spatial information.              #
        # 3. Pass the sequence through the Transformer encoder.                   #
        # 4. Average pool patch vectors to get a feature vector for each image.   #
        #    You may find torch.mean useful.                                      #
        # 5. Feed it through a linear layer to produce class logits.              #
        ############################################################################
        # [中文] TODO:
        # 执行前向传播愿景转换器。1。将输入图像转换成补丁矢量序列。2。添加位置编码以保留空间信息。3。通过转换器编码器传送序列。4。平均的池状补丁矢量为每个图像获取一个特征向量。您可能会发现torch.mean有用
        #。5。通过线性层喂养它,以生成类登录。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束


        return logits
