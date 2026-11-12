import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.


    中文对照：
    描述RNN 使用一个经常性神经网络,从图像特征中生成字幕。

    RNN接收大小为D的输入矢量, vocab 大小为 V, 工作长度为 T 的序列, 隐藏 H 的 RNN 尺寸, 使用尺寸为 W 的单词矢量, 并在大小为 N 的小型信箱上操作。

    注意,我们不使用 任何正则化为描述RNN。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.


        中文对照：
        构造一个新的描述RNN 实例。

        投入: -word_to_idx:提供词汇的字典。它包含 V 项,并将每个字符串映射为区域中唯一的整数[0, V). -  input_dim:输入图像特征向量的尺寸 D。 - wordvec_dim:字矢量的维度 W。 - hidden_dim隐藏的 RNN 状态的维度 H。 - cell_type:使用何种RNN;'rnn'或'lstm'。 - dtype:要使用的纽扣数据类型;使用浮点32进行训练,用浮点64进行数字梯度检查。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        # [中文] 初始化单词矢量
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        # [中文] 初始化 CNN - > 隐藏状态投影参数
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # Initialize parameters for the RNN
        # [中文] 初始化 RNN 参数
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        # [中文] 初始化输出为 vocab 权重
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # Cast parameters to correct dtype
        # [中文] 用于校正 dtype 的播种参数
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss


        中文对照：
        计算 RNN 的训练时间损失。 我们输入这些图像的图像特征和地面真相说明, 并使用 RNN (or LSTM) 来计算所有参数的损失和梯度。

        输入: - 特征: 输入图像特征, 形状(N, D) - 说明: 地面真实说明; 形状的整数数组( N, T + 1 ), 其中每个元素都位于 0 y[i, t] < V

        返回: - 损失: 标量 损失
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # [中文] 将描述切成两块: captions_in 拥有除最后一个单词之外的所有内容, 并将输入到 RNN; captions_out
        # 拥有除了第一个单词之外的一切内容, 这就是我们期待 RNN 生成的内容。 这些内容被相对部分抵消, 因为 RNN 在接收单词 t 后应该生成单词( t+ 1
        # )。 captions_in 的第一个元素将是 START 标语, 而第一个单词是 captions_out。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        # [中文] 你会需要这个的
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        # [中文] 角的权重和偏差从图像特性转换为初始隐藏状态
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        # [中文] 单词嵌入矩阵
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        # [中文] 输入隐藏, 隐藏隐藏隐藏, 以及对 RNN 的偏向
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        # [中文] 隐藏到蒸汽变形的权重和偏差
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.                  #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #       
        # Please ensure that your implementation is agnostic of the input tensors  #
        # data types.                                                              #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # You also don't have to implement the backward pass.                      #
        ############################################################################
        # [中文] TODO: 在 描述 RNN 中执行 前向传播。 在 前向传播 中, 您需要做以下工作: (1) 使用折线转换来从图像特性中计算初始隐藏状态。
        # 这应该产生形状 (N, H) (2) 的数组, 使用单词嵌入层将单词从指数 captions_in 转换为矢量, 给出形状的数组 (N, T, W)。
        # (3) 使用香草 RNN或 LSTM (取决于 self.cell_type ) 来处理输入单向量的序列, 并生成所有时间步骤的隐藏状态矢量, 而生成形状
        # (N, T, H) 的数列。 (4) 使用 (temporal) 的线段转换来计算分数。 使用隐藏状态, 给出形状数组的词汇(N, T, V) .
        # (5)使用(temporal)用于计算损失的软最大值captions_out,忽略输出单词所在的点<NULL>使用上面的掩码。 请确保您的执行对输入
        # shors 数据类型具有不可知性。 不要担心权重或梯度的正规化。 您也不必执行反向传播 .

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        return loss

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.


        中文对照：
        为模型运行一个测试时间 前向传播,为输入特征向量取样说明。

        每次时间步骤,我们都会将当前单词和先前的隐藏状态嵌入 RNN, 以获得下一个隐藏状态, 使用隐藏状态来获得所有 vocab 单词的评分, 并选择得分最高的单词为下一个单词。 初始隐藏状态是通过对输入图像特性应用折线转换来计算, 初始单词是 <START> 符号。

        对于 LSTMS, 您也必须跟踪单元格状态; 在这种情况下, 初始单元格状态应该是零。

        输入: - 特性: 形状 (N, D) 的输入图像特性的矩阵。 - max_length: 生成的字幕的最大长度 T。

        返回: - 描述: 形状的矩阵(N, max_length)给出样本描述,其中每个元素在范围中为整数 [0描述的第一个要素应该是第一个抽样词,而不是<START>象征。
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # Unpack parameters
        # [中文] 卸载参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # [中文] TODO: 执行模型的测试- 时间取样。 您需要将所学的松动转换到输入图像特性中, 从而初始化 RNN 的隐藏状态。 您输入给 RNN的第一个单词应该是
        # <START> 符号; 其值存储在变量 self._start 中。 每次时间步骤您需要做到: (1) 使用所学的单词嵌入 (2) 来嵌入上前一个单词
        #, 使用先前的隐藏状态和嵌入的当前单词来获取下一个隐藏状态。 (3) 将所学过的松动转换应用到下一个隐藏状态, 以获得词汇 (4) 中所有单词的分数 选择
        # 以最高分的单词为下一个单词, 写入它(the word index)简单化,您不需要在<END>符号是抽样的, 但如果您想要, 您可以使用。 HINT:
        # 您将无法使用rnn_forward或lstm_forward函数;
        # 您需要调用rnn_step_forward或lstm_step_forward循环。注意:我们仍在此函数中用微型插头工作。如果您正在使用 LSTM
        #,请将第一个单元格状态初始化为零。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        return captions
