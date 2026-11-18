"""This file defines layer types that are commonly used for recurrent neural networks.


中文对照：
此文件定义循环神经网络常用的各类层。
"""
import torch


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)


    中文对照：
    仿射层的前向传播。将每个样本展平为 D 维向量，再用形状 (D,M) 的权重及 (M,) 的偏置映射到 M 维；输出 out 为 (N,M)。
    """
    out = x.reshape(x.shape[0], -1) @ w + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)


    中文对照：
    使用 tanh 激活的普通 RNN 单步前向传播。x 为 (N,D)，prev_h 为 (N,H)，Wx 为 (D,H)，Wh 为 (H,H)，b 为 (H,)；返回下一隐藏状态 next_h（N,H）。
    """
    next_h = None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN.                 #
    ##############################################################################
    # [中文] 待完成：实现普通 RNN 单个时间步的前向传播。
    x_h = x.mm(Wx)
    prev_h = prev_h.mm(Wh)
    next_h = torch.tanh(x_h + prev_h + b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.

    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)


    中文对照：
    普通 RNN 在完整序列上的前向传播。输入为 N 条长度为 T、每步 D 维的序列，隐藏维度为 H；结合初始隐藏状态逐时间步计算，返回各时间步隐藏状态。
    """
    h = None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # [中文] 待完成：实现普通 RNN 对整段输入序列的前向传播。复用上面定义的 rnn_step_forward，可以使用 for 循环遍历时间步。
    N, T, D = x.shape

    prev_h = h0
    hidden_states = []

    for t in range(T):
        prev_h = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        hidden_states.append(prev_h)

    h = torch.stack(hidden_states, dim=1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    return h


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.


    中文对照：
    词嵌入前向传播。x 为 (N,T) 的整数词索引，取值在 [0,V)；W 为 (V,D) 的词向量矩阵。返回 out（N,T,D），为序列中各个词对应的向量。
    """
    out = None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using Pytorch's array indexing.         #
    ##############################################################################
    # [中文] 待完成：实现词嵌入的前向传播。原题提示：使用 PyTorch 的数组索引，一行即可完成。
    out = W[x]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)


    中文对照：
    LSTM 单步前向传播。x 为 (N,D)，prev_h、prev_c 为 (N,H)，Wx 为 (D,4H)，Wh 为 (H,4H)，b 为 (4H,)；返回 next_h 和 next_c，形状均为 (N,H)。文件已提供 sigmoid 函数。
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # [中文] 待完成：实现 LSTM 单个时间步的前向传播，可使用上面数值稳定的 sigmoid 函数。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。

    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.

    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)


    中文对照：
    LSTM 在完整序列上的前向传播。对 N 条长度为 T 的序列逐步更新隐藏状态和记忆单元状态，返回各时间步的隐藏状态；参数含义见英文列表。
    """
    h = None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # [中文] 待完成：实现 LSTM 对整个时间序列的前向传播，复用刚刚实现的 lstm_step_forward。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 自行实现的代码区域结束。

    return h


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.

    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)


    中文对照：
    时间序列仿射层的前向传播。对 x（N,T,D）中每个时间步的 D 维向量应用同一仿射变换，w 为 (D,M)，b 为 (M,)，输出 out 为 (N,T,M)。
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = (x.reshape(N * T, D) @ w).reshape(N, T, M) + b
    return out


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.

    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss


    中文对照：
    RNN 使用的时间序列 Softmax 损失。x（N,T,V）给出每个时间步的词表评分，y（N,T）给出真实词索引。在各时间步计算交叉熵、求和，再按批大小 N 平均。mask（N,T）指定哪些位置计入损失，从而忽略用于补齐序列的 NULL 标记。返回标量 loss。
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T).long()
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
