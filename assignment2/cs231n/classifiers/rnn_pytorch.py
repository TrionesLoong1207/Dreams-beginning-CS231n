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
    根据图像特征用 RNN 生成描述。图像特征维度为 D，词表大小为 V，词向量维度为 W，隐藏状态维度为 H。这里的模型类负责计算，不负责完整优化过程。
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
        构造 CaptioningRNN。word_to_idx 将 V 个词映射为 [0,V) 中的唯一整数；input_dim、wordvec_dim、hidden_dim 分别是图像特征、词向量、隐藏状态维度。cell_type 选择 rnn 或 lstm；训练可用 float32，数值梯度检查建议用 float64。
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
        # [中文] 初始化词向量。
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        # [中文] 初始化从 CNN 图像特征到初始隐藏状态的投影参数。
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # Initialize parameters for the RNN
        # [中文] 初始化 RNN 参数。
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        # [中文] 初始化从隐藏状态映射到词表评分的权重。
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # Cast parameters to correct dtype
        # [中文] 将参数转换为所需的数据类型。
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
        计算 RNN 图像描述模型的训练损失。输入为图像特征和真实描述，通过 RNN 或 LSTM 对词序列建模；具体输入形状、返回值见英文约定。
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # [中文] 将描述序列错开一位：captions_in 去掉最后一个词，captions_out 去掉第一个词。模型读入第 t 个词后应预测第 t+1 个词。captions_in
        # 的开头为 START 标记，captions_out 的开头为描述中的第一个词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        # [中文] 后面会用到这个变量。
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        # [中文] 将图像特征仿射映射为初始隐藏状态的权重和偏置。
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        # [中文] 词嵌入矩阵。
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        # [中文] RNN 的输入到隐藏状态权重、隐藏状态到隐藏状态权重及偏置。
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        # [中文] 从隐藏状态映射到词表评分的权重和偏置。
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
        # [中文] 待完成：实现 CaptioningRNN 的前向传播。依次把图像特征映射为初始隐藏状态 (N,H)，将输入词索引嵌入为词向量 (N,T,W)，按 self.cell_type
        # 用 RNN 或 LSTM 得到各时间步隐藏状态 (N,T,H)，再映射为词表评分 (N,T,V)。用 captions_out 计算时间序列 Softmax 损失，并用给定
        # mask 忽略 NULL 位置。实现应兼容输入张量的数据类型；本题无需加权重正则化，也不要求手写反向传播。
        h0 = affine_forward(features, W_proj, b_proj)
        word_vectors = word_embedding_forward(captions_in, W_embed)
        hidden_states = rnn_forward(word_vectors, h0, Wx, Wh, b)
        scores = temporal_affine_forward(hidden_states, W_vocab, b_vocab)
        loss = temporal_softmax_loss(scores, captions_out, mask)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

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
        测试时从图像特征逐步生成描述。每步嵌入当前词、更新循环状态、计算词表评分，再选择下一词。先用图像特征初始化状态，以 START 标记开始；返回生成的词索引序列。
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # Unpack parameters
        # [中文] 解包参数。
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
        # [中文] 待完成：实现测试时的描述生成。由图像特征计算初始隐藏状态，以 self._start 对应的 START 标记作为首个输入。每个时间步依次执行词嵌入、RNN/LSTM
        # 单步前向传播、词表评分，再选择评分最高的词，将词索引写入 captions。无需强制在 END 标记处停止。原题提示：这里要循环调用单步函数，不能直接使用处理整段序列的
        # rnn_forward/lstm_forward；仍需按小批量处理。使用 LSTM 时，初始记忆单元状态置零。
        h0 = affine_forward(features, W_proj, b_proj)
        current_word = torch.full(
            (N,), self._start, dtype=torch.long, device=features.device
        )

        for t in range(max_length):
            word_vector = word_embedding_forward(current_word, W_embed)
            prev_h = rnn_step_forward(word_vector, h0, Wx, Wh, b)
            scores = affine_forward(prev_h, W_vocab, b_vocab)
            current_word = torch.argmax(scores, dim=1)
            captions[:, t] = current_word.to(captions.device)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return captions
