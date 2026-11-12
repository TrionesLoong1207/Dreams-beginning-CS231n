"""This file defines layer types that are commonly used for recurrent neural networks.


中文对照：
此文件定义经常神经网络常用的图层类型。
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
    计算角线 前向传播 层的 (fully connected)。

    输入 x 有形状 (N,d_1, ...,  d_k),并包含一小批N实例,其中每个实例x[i]有形状(d_1, ...,  d_k我们将将每个输入重塑为维度矢量D=d_1  * ... *  d_k,然后将它转换为维度M的输出矢量。

    输入: - x: 含有输入数据的火炬数组,形状(N,d_1, ...,  d_k) - w:重力、形状的火炬数组(D, M)- b:一束偏差,形状的火炬(M,)

    返回图普: - 退出: 输出, 形状 (N, M)
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
    使用 tanh 激活 功能运行香草 RNN 的单时段 前向传播。

    输入数据有D维度, 隐藏状态有H维度, 而小批量为 N 大小。

    输入: - x: 此时间步骤的输入数据, 形状 (N, D) - prev_h: 先前时间步骤的隐藏状态, 形状(N, H) - Wx: 输入到隐藏连接的权重矩阵, 形状为 (D, H) - Wh: 隐藏到隐藏连接、 形状为(H, H) - b: 形状 (H,) 的比喻

    返回一个图普: - next_h: 形状 (N, H) 的下一个隐藏状态
    """
    next_h = None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN.                 #
    ##############################################################################
    # [中文] 为香草RNN执行一个前进步骤。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
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
    运行一个香草 RNN 前进 整个数据序列。

    我们假设一个由 T 矢量构成的输入序列, 每个维度为 D。 RNN 使用隐藏的 H 大小, 并且我们在一个包含 N 序列的小型批次上工作。 在运行 RNN 前行后, 我们返回所有时间步骤的隐藏状态。

    输入数: - x: 形状整个时间序列的输入数据(N, T, D)- h0: 形状的初始隐藏状态(N, H)- Wx:输入到隐藏连接的权重矩阵,形状(D, H)- WH: 隐藏到隐藏连接的权重矩阵, 形状(H, H)- b: 形状的两边形(H,)

    返回: - h: 形状 (N, T, H) 整个时间序列的隐藏状态
    """
    h = None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # [中文] TODO: 执行前向传播对于在输入数据序列上运行的香草 RNN。您应该使用rnn_step_forward函数。您可以使用循环来帮助计算前向传播 .

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
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
     单词嵌入 前向传播。

    我们用N大小的小型小桶操作, 每一个序列都有T长度。 我们假设一个V字的词汇, 将每个单词指定为 D 的矢量。

    输入: - x: 形状 (N, T) 的整数数组给出单词索引。 每个元素 x must 的 idx 都在 0 \\ idx < V. - W: 形状(V, D) 的权重矩阵中, 给所有单词矢量。

    返回一个 tuple: - out: 形状 (N, T, D) 的矩阵,为所有输入单词提供单词矢量。
    """
    out = None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using Pytorch's array indexing.         #
    ##############################################################################
    # [中文] TODO: 对单词嵌入执行 前向传播。 HINT: 可以用 Pytoch 的数组索引在一个行中完成。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束
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
     LSTM 的单时段 前向传播。

    输入数据有D维度, 隐藏状态有H维度, 我们使用一个小批量大小的 N。

    请注意, 此文件中已经为您提供了 sigmoid () 函数。

    输入: - x: 输入数据,形状 (N, D) - prev_h: 先前隐藏状态, 形状 (N, H) - prev_c: 前单元格状态, 形狀 (N, H) - Wx: 输入到隐藏的权重, 形状(D, 4H) - Wh: 隐藏到隐藏的重, 形状为 (H, 4H) - b: 形状为(4H,)

    返回一个图普: - next_h: 下一个隐藏状态, 形状 (N, H) - next_c: 另一个单元格状态, 特性 (N, H)
    """
    next_h, next_c = None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # [中文] TODO: 对 LSTM 的单个时间步执行 前向传播。 您可能想要使用上面的数值稳定的 Sigmoid 执行。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束

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
     LSTM 覆盖整个数据序列的前向传播。

    我们假设一个由 T 矢量组成的输入序列, 每个维度为 D。 LSTM 使用的是隐藏的 H 大小, 我们在一个包含 N 序列的小型批次上工作。 在运行 LSTM 前行后, 我们返回所有时间步骤的隐藏状态。

    注意初始单元格状态作为输入通过, 但初始单元格状态设为零。 另请注意, 单元格状态未返回; 它是 LSTM 的内部变量, 且无法从外部访问。

    输入数: - x: 形状输入数据(N, T, D)- h0: 初始隐藏形状状态(N, H)- Wx:输入到隐藏连接的权重,形状(D, 4H)- WH: 隐藏到隐藏连接的权重, 形状(H, 4H)- b: 形状的两边形(4H,)

    返回: - h: 形状(N, T, H) 所有序列的所有时间步骤的隐藏状态
    """
    h = None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # [中文] TODO: 执行前向传播整个时间序列的 LSTM。 您应该使用 lstm_step_forward函数。

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    # [中文] 代码结束

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
     时间角层 前向传播。

    输入是一组D- 维矢量, 排列成一小批N时间序列, 每组长度 T。 我们使用一个折线函数, 将这些矢量转换成一个新的维度 M 矢量。

    输入数: - x: 形状输入数据(N, T, D)- w: 形状的权重(D, M)- b: 形状的两边形(M,)

    返回形状 (N, T, M) 的图普: - 退出: 输出数据
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
    用于 RNN 的软负损失的时间版本。

    我们假设我们正在对长度为T的时序的每个时序的V大小词汇进行预测。 输入 x 给出所有时序词汇元素的分数, y 给出每个时程的地面真实元素的指数。 我们在每个时序使用跨轴损失, 在所有时序中进行总结, 并平均横跨时序进行计算。

    作为一种额外的复杂因素,我们也许想在某些时间步骤中忽略模型输出,因为不同长度的序列可能已经合并成一个小批和加了NULL标志。 任择掩码参数告诉我们哪些元素应该促成损失。

    输入: - x: 输入分数,形状(N, T, V) - y: 形状 (N, T) 的地面真象指数,其中每个元素都位于 0 范围 y[i, t] < V - 掩码: 形状(N, T) 的布尔恩数组,其中 mask[i, t] 表示在形状 x[i, t] 上的分数是否应该促成损失。

    返回: - 损失: 标量 给付损失
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
