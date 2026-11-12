from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)


    中文对照：
    计算角线 前向传播 层的 (fully connected)。

    输入 x 有形状 (N,d_1, ...,  d_k),并包含一小批N实例,其中每个实例x[i]有形状(d_1, ...,  d_k我们将将每个输入重塑为维度矢量D=d_1  * ... *  d_k,然后将它转换为维度M的输出矢量。

    输入数组: - x: 含有输入数据、形状( N, d_1,..., d_k ) - w: 权重、形状 (D, M) - b: 形状 (M,) 偏差、形状偏差的缩略数组

    返回一个图普: - 退出: 输出, 形状 (N, M) - 缓存: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # [中文] TODO: 从任务 1 复制您的解决方案。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)


    中文对照：
    计算角线 反向传播 层的 (fully connected)。

    输入: - dout: 上流衍生物,形状 (N, M) - 缓存: 图例: - x: 输入数据,形状 (N, d_1,... d_k ) - w: 权重,形状 (D, M) - b: 比斯, 形状 (M,)

    返回图普的: - dx: 形状 x 的梯度( N, d1,..., d_k ) - dw: 形状 (D, M) 的瓦的梯度 - db: 形状 b的梯度 (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # [中文] TODO: 从任务 1 复制您的解决方案。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x


    中文对照：
    计算前向传播直线单位的层(ReLUs) .

    输入: - x: 任何形状的输入

    返回一个图普: - 退出: 输出, 形状与 x - 缓存: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # [中文] TODO: 从任务 1 复制您的解决方案。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x


    中文对照：
    计算反向传播直线单位的层(ReLUs) .

    输入: - dout: 任何形状的上流衍生物 - 缓存: 输入 x, 与 dout 相同形状

    返回: - dx: 相对于 x 的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # [中文] TODO: 从任务 1 复制您的解决方案。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x


    中文对照：
    计算软负分级的损失和梯度。

    输入: - x: 输入数据, 形状 (N, C) 的输入数据, 其中 x[i, j] 是 第 i 个 输入的 第 j 个 类的分数。 - y: 标签的矢量, 形状(N,) 的分数, 其中 y[i] 是 x[i] 和 0 y[i] < C > 的标签

    返回: - 损失: 标量 提供损失 - dx: x 损失的梯度
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # [中文] TODO: 从任务 1 复制您的解决方案。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass


    中文对照：
     批量归一化 前向传播。

    在训练过程中,样本平均值和(uncorrected)样本差异从小批统计中计算出来,并用于将收到的数据归一化。在训练过程中,我们还保持每个特性的平均值和差异的指数衰减平均值,这些平均值用于测试时的数据归一化。

    每次时间步骤,我们都会使用基于动量参数的指数衰减来更新运行中的平均值和差数:

     running_mean = 动量 * running_mean + ( 1 - 动量) * sample_mean running_var = 动量 * running_var + (1 -动量) *sample_var

    请注意批量解析文件显示一种不同的测试时间行为:它们使用大量训练图像而不是运行平均值来计算每个特性的样本平均值和差异。 对此我们选择使用运行平均值, 因为它们不需要额外的估计步骤; 批次解析的火炬7 实施也使用运行平均值。

    输入: - x: 形状 (N, D) - 伽玛的数据: 形状(D,) - 贝塔: 形状 > (D,) - bn_param 的 Shift 参数: 形状有以下键的字典: - 模式: 'train' 或 'test'; 需要 - eps: 数字稳定性的常数 - 动量: 运行平均/ 差异的常数。 - running_mean: 形状的弧 (D,), 显示特征的运行平均值 - running_var 形状 (D,) 的直径, 特征的变化变化。

    返回图普: - out: 形状 (N, D) - 缓存: 反向传播 中需要的图普
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # [中文] TODO: 在批量规范中执行训练时间 前向传播。 使用迷你批量统计来计算平均值和差异, 使用这些统计数据来将接收的数据正常化,
        # 并使用伽马和贝塔来缩放和转换正统数据。 您应该将输出存储在变量中。 您需要的 反向传播 中的任何中间点应该存储在缓存变量中。 同时,
        # 您还应该使用您计算出的样本平均值和差异与动量变量一起更新运行中平均值和变化中的差异, 将结果存储在 running_mean 和 running_var
        # 变量中。 注意, 尽管您应该跟踪运行中的差异, 您应该使用您所计算的样本平均值和变量来更新运行中的平均值和差异。 应使基于标准差 (square root
        # of variance) 的数据正常化。 引用原始论文( https://arxiv.org/abs/1502.03167) )可能会有帮助。
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        # [中文] 代码结束
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # [中文] TODO: 执行批量归一化的测试时间 前向传播。 使用运行中的平均值和差异来将接收的数据归一化, 然后使用伽马和贝塔来缩放和转换已归正的数据。
        # 将结果保存在外变量中。
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # [中文] 代码结束
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    # [中文] 将更新的运行手段存储到 bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)


    中文对照：
     批量归一化 反向传播。

    为此,您应在论文上写出批量归一化的计算图,并通过中间节点向后传播梯度。

    输入: - dout: 上流衍生物,形状 (N, D) - 缓存: batchnorm_forward 的中间体变量。

    返回: - dx: 相对于输入 x 的梯度, 形状 (N, D) - dgamma: 相对于比例参数伽马, 形状(D,) - dbeta: 相对于形状 (D,) 的移动参数贝塔的梯度
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # [中文] TODO: 执行批量归一化的 反向传播。 保存 dx、 dgamma 和 dbeta 变量中的结果。 引用原始论文 (
    # https://arxiv.org/abs/1502.03167) ) 可能会很有帮助。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward


    中文对照：
    批量归一化的替代品 反向传播。

    您应该为论文上的批量正态 反向传播 绘制衍生物, 并尽可能简化。 您应该能够为 反向传播 得出一个简单的表达式。 请查看 juppyter 笔记本以获得更多提示。

    注意: 此执行应预期收到与 batchnorm_backward 相同的缓存变量, 但可能不会使用缓存中的全部值。

    输入/输出:与 batchnorm_backward 相同
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # [中文] TODO: 在批量解析中执行 反向传播。 保存 dx、 dgamma 和 dbeta 变量的结果。 在计算中心输入的梯度后,
    # 您应该能够在单个语句中计算投入的梯度; 我们的应用程序符合单条 80 - 字符直线。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass


    中文对照：
     分层归一化 前向传播。

    在训练期间和测试期间,收到的数据按数据点标准化,然后按与批次归一化相同的伽马和贝塔参数进行缩放。

    请注意,与分批归一化相比,列车期间的行为和分层归一化测试时的行为是相同的,我们不需要跟踪任何类型的平均运行量。

    输入: - x: 形状 (N, D) - 伽玛的数据: 形状(D,) - 贝塔: 形状 > (D,) - ln_param 的 Shift 参数: 字典中包含以下键: - eps: 数字稳定性的常数

    返回图普: - out: 形状 (N, D) - 缓存: 反向传播 中需要的图普
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # [中文] TODO: 对层规范执行训练时间 前向传播。 将收到的数据标准化, 并用伽马和贝塔来缩放和转换正常数据。 HINT:
    # 可以通过略微修改批量归一化的训练时间来做到这一点, 并插入一行或两行正确位置的代码。 特别是, 您能否想到您可以进行的任何矩阵变换,
    # 使您能够复制批量规范代码并保持几乎不变 ?

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)


    中文对照：
     分层归一化 反向传播。

    对于这个执行,你可以在很大程度上依赖你已经做的工作 批量归一化。

    输入: - dout: 上流衍生物,形状 (N, D) - 缓存: layernorm_forward 的中间体变量。

    返回: - dx: 相对于输入 x 的梯度, 形状 (N, D) - dgamma: 相对于比例参数伽马, 形状(D,) - dbeta: 相对于形状 (D,) 的移动参数贝塔的梯度
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # [中文] TODO: 执行反向传播HINT: 可以通过略微修改您的批量归一化的训练时间来做到这一点。前向传播仍在申请!

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.


    中文对照：
     倒数 前向传播 的 Dropout（随机失活）。

    请注意,这与香草版本 Dropout（随机失活） 不同。这里,p是保持神经元输出的概率,而不是降低神经元输出的可能性。详情见http://cs231n.github.io/neural-networks-2/#reg。

    输入: - x: 输入数据, 任何形状 - dropout_param: 字典, 键如下: - p: Dropout（随机失活） 参数。 我们保留每个神经元输出的概率 p. - 模式: 'test' 或 'train'。 如果模式是火车, 那么执行 Dropout（随机失活）; 如果模式是测试, 那么返回输入。 - 种子: 随机数字生成器的种子。 传递种子使此函数具有确定性, 这在梯度检查中是必需的, 但在真正的网络中不是。

    输出: - 输出: 与 x 相同形状的数组。 - 缓存: tuple (dropout_param, mask)。 在训练模式中, 掩码是用来乘进数的 Dropout（随机失活） 掩码; 在测试模式中, Mask is None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # [中文] TODO: 对倒转 Dropout（随机失活） 执行 前向传播 训练阶段。 在掩码变量中保存 Dropout（随机失活） 掩码。
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        # [中文] 代码结束
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # [中文] TODO: 对倒数 Dropout（随机失活） 执行测试阶段 前向传播。
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # [中文] 代码结束

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.


    中文对照：
     倒数 反向传播 的 Dropout（随机失活）。

    输入: - dout: 任何形状的上流衍生物 - 缓存: (dropout_param, mask) from dropout_forward。
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # [中文] TODO: 实施训练阶段反向传播倒转Dropout（随机失活）
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # [中文] 代码结束
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)


    中文对照：
    对革命层的朴素实现 前向传播。

    输入由 N 数据点组成, 每个有 C 频道、 H 高度 和 w 宽度。 我们用 F 不同的滤波器将每个输入相混合, 每个滤波器横跨所有 C 频道, 高 HH 和宽 WW。

    输入: - x: 形状(N, C, H, W) - w: 形状 (F, C, HH, WW) - b: 形状的滤波器权重 (F,) - conv_param: 字典,其键如下: - 'stride': 水平和垂直方向中相邻感受野之间的像素数量。 - 'pad': 输入时将使用像素数量。

    在铺垫期间,'pad'零应该以对称方式排列(A/C.5/49/70,附件)。i.e在输入的高度和宽度轴沿线。 注意不要直接调整原始输入 x。

    返回: - 输出: 输出数据, 形状( N、 F、 H、 W ), H' 和 W' 是 H 的给定值 =1  + (H +  2* 垫子 - HH) / 踏步 W' =1  + (W +  2* 垫子 - WW) / 步态 - 缓存:(x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # [中文] TODO: 执行递归 前向传播。 提示: 您可以使用函数 np.pad 来粘贴。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b


    中文对照：
    对革命层的朴素实现 反向传播。

    输入: - dout: 上流衍生物。 - 缓存: 一图普(x, w, b, conv_param)以年数计( 单位: 单位: 百分比:%: 0): 1, 1, 2, 3, 4, 6, 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 28, 24, 27, 26, 25, 20, 30, 31, 29, 15, 62, 69, 61, 70, 76, 71, 81, 86, 87, 89, 80, 84, 73, 85, 83, 96, . 10, 103, 中conv_forward_naive

    返回: - dx: x - dw 的梯度: w - db 的渐變: b 的渐梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # [中文] TODO: 执行革命性 反向传播。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)


    中文对照：
    最大池化层的 前向传播 的朴素实现。

    输入: - x: 输入数据,形状 (N, C, H, W) - pool_param:字典,键如下: - 'pool_height':每个集合区域的高度 - 'pool_width': 每个集合区域的宽度 - 'stride':邻近集合区域的距离

    此处不需要粘贴, 如您可以假设: - (H - pool_height )% ride = 0 - (W - pool_width ) % ride= 0

    返回: - 退出:输出数据,形状(N、C、H、W),H'和W由 H = 1 + (H - pool_height) / stride W =1 +(W - pool_width )/stride - 缓存:(x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # [中文] TODO: 执行最大共享 前向传播

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x


    中文对照：
    最大池化层的 反向传播 的朴素实现。

    输入: - dout: 上流衍生物 - 缓存: (x, pool_param) 的图普,如 前向传播。

    返回: - dx: 相对于 x 的梯度
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # [中文] TODO: 执行最大共享 反向传播

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass


    中文对照：
    计算空间批量归一化的前向传播。

    输入数: - x: 形状输入数据(N, C, H, W)- 伽马: 比例参数, 形状(C,)- beta: 变换参数, 形状(C,)  -  bn_param: 字典中包含以下键: - 模式:'train'或'test'; 所需的 - eps: 数字稳定性的常数 - 动量: 运行均值/方差的常量。 动量=0意思是,老信息在每次步骤中都完全被丢弃,而动量=1表示新信息从未被包含。 默认动量 =0.9- 在多数情况下应该能很好地工作。running_mean: 形状的数组(D,)给出运行特征的平均值 - running_var形状数组(D,)正在运行 特征差异

    返回一个图普: - 退出: 输出数据, 形状 (N, C, H, W) - 缓存: 反向传播 需要的值
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # [中文] TODO: 在空间批量归一化中执行 前向传播。 HINT: 您可以使用上面所执行的批量归一化的香草版本来实施空间批次归一化。
    # 您的批次归一化执行应该非常短; 我们的行小于五行。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)


    中文对照：
    计算空间批量归一化的反向传播。

    输入: - dout: 上流衍生物,形状 (N, C, H, W) - 缓存: 前向传播 的值

    返回: - dx: 输入的梯度, 形状 (N, C, H, W) - dgamma: 形状 (C,) - dbeta: 形状(C,) 的梯度
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # [中文] TODO: 在空间批量归一化中执行 反向传播。 HINT: 您可以使用上面所执行的批量归一化的香草版本来实施空间批次归一化。
    # 您的批次归一化执行应该非常短; 我们的行小于五行。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass


    中文对照：
    计算空间组归一化的前向传播。

    与层级归一化相反,组级归一化将数据中的每个条目分割为G相毗连部分,然后独立地实现。 然后,对数据采用与批次归一化和层归一化相同的方式,逐个调整和缩放。

    输入: - x: 形状 (N, C, H, W) - 伽马: 比例参数, 形状 (1, C, 1, 1) - 贝塔: Shift 参数, 元件 (1, C, 1, 1) - G: 要拆分的组的整数朗伯, 应该是 C - gn_param 的三角形: 字典, 键如下: - eps: 数字稳定性的常数

    返回一个图普: - 退出: 输出数据, 形状 (N, C, H, W) - 缓存: 反向传播 需要的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # [中文] TODO: 执行空间组常规的 前向传播。 这将与层规范的实施非常相似。 特别是, 想想如何转换矩阵, 使大部分代码既与训练时间的批次归一化相似,
    # 也与层归一化相似 !

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)


    中文对照：
    计算空间组归一化的反向传播。

    输入: - dout: 上流衍生物,形状 (N, C, H, W) - 缓存: 前向传播 的值

    返回: - dx: 输入的梯度, 形状 (N, C, H, W) - dgamma: 形状 (1, C, 1, 1) - dbeta: 形状(1, C, 1, 1) 的梯度
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # [中文] TODO: 执行空间群落归一化 反向传播。 这将与层规范执行极为相似。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    return dx, dgamma, dbeta
