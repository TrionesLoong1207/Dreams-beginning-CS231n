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
    用 Transformer 解码器根据图像特征生成文本描述。D 是图像特征维度，V 是词表大小，W 是词嵌入维度；该类定义模型计算，训练由独立训练器管理。
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
        构造 CaptioningTransformer。word_to_idx 将词映射到 [0,V)；input_dim 为图像特征维度，wordvec_dim 为词嵌入维度，num_heads 为头数，num_layers 为 Transformer 层数，max_length 为最大序列长度。
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
        初始化网络权重。
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
        给定图像特征 features（N,D）和完整真实描述 captions（N,T），计算每个时间步的词表评分 scores（N,T,V）。由于训练时一次传入整段序列，必须屏蔽未来时间步，避免提前看到后面的词。
        """
        N, T = captions.shape
        # Create a placeholder, to be overwritten by your code below.
        # [中文] 创建占位值，后续由你实现的代码覆盖。
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
        # [中文] 待完成：实现图像描述 Transformer 的前向传播。原题提示：先对描述序列做词嵌入并添加位置编码，将图像特征投影到同一嵌入维度；再构造
        # tgt_mask，屏蔽未来时间步，可参考 torch.tril()；最后将文本、图像嵌入与掩码送入解码器，并将输出投影成各 token 的词表评分。
        captions_emb = self.embedding(captions)
        captions_emb = self.positional_encoding(captions_emb)
        features_emb = self.visual_projection(features).unsqueeze(1)

        tgt_mask = torch.tril(
            torch.ones(T, T, device=captions.device)
        )

        output = self.transformer(
            tgt=captions_emb,
            memory=features_emb,
            tgt_mask=tgt_mask
        )

        scores = self.output(output)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

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
        给定图像特征，用贪心解码生成描述，每个时间步选择当前评分最高的 token。features 为 (N,D)，max_length 控制最大生成长度；返回词索引序列。
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            # [中文] 创建空的 captions 张量，初始所有位置为 NULL 标记。
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            # [中文] 创建只含开始标记的初始描述序列。
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                # [中文] 仅预测下一个 token，忽略其他时间步的预测。
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                # [中文] 从词表中选择预测概率最大的词索引。
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                # [中文] 更新完整描述及当前部分描述序列。
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
    Vision Transformer（ViT）的实现。
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
        初始化 ViT。img_size、patch_size 为正方形图片与图像块边长，in_channels 为通道数，embed_dim 为图像块嵌入维度；num_layers、num_heads 为编码器层数和头数，dim_feedforward 为前馈网络隐藏维度，num_classes 为类别数，dropout 为随机失活概率。
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer to predict class scores from pooled token.
        # [中文] 最终分类层根据池化后的特征生成各类别评分。
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.


        中文对照：
        初始化网络权重。
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
        ViT 前向传播。输入 x 为 (N,C,H,W) 的图片张量，输出 logits 为每张图片的类别评分。
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
        # [中文] 待完成：实现 ViT 前向传播。把图像转换为图像块向量序列，添加位置编码，经 Transformer
        # 编码器处理后，对图像块向量做平均池化，得到每张图像的特征，再用线性层输出类别 logits。原题提示：平均池化可参考 torch.mean。
        x_patches = self.patch_embed(x)
        x_pos = self.positional_encoding(x_patches)
        x_trans = self.transformer(x_pos)
        x_pool = x_trans.mean(dim=1)
        logits = self.head(x_pool)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。


        return logits
