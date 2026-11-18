from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

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
    仿射（全连接）层的前向传播。输入 x 的形状为 (N,d_1,...,d_k)，先将每个样本展平成 D=d_1×...×d_k 维向量，再映射到 M 维。输入权重 w 为 (D,M)，偏置 b 为 (M,)；返回 out，形状 (N,M)，以及 cache=(x,w,b)。
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # [中文] 待完成：实现仿射层的前向传播，将输入样本展平成行向量后计算，结果存入 out。
    x_ = x.reshape(x.shape[0], -1)
    x_ = x_.dot(w)
    out = x_ + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

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
    仿射层的反向传播。dout 是上游梯度，形状为 (N,M)；cache 保存前向传播的 x、w、b。返回损失对输入、权重和偏置的梯度 dx、dw、db，形状分别与 x、w、b 相同。
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # [中文] 待完成：实现仿射层的反向传播。
    x_rows = x.reshape(x.shape[0], -1)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_rows.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x


    中文对照：
    ReLU 激活层的前向传播。输入 x 可以是任意形状；返回同形状的 out，并将 x 缓存在 cache 中。
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # [中文] 待完成：实现 ReLU imum的前向传播。
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x


    中文对照：
    ReLU 激活层的反向传播。dout 是上游梯度，cache 是前向传播保存的 x，两者形状相同；返回损失对 x 的梯度 dx。
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # [中文] 待完成：实现 ReLU 的反向传播。
    dx = dout * (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

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
    批归一化的前向传播。训练时用当前小批量的均值和未作无偏修正的方差归一化，并用 momentum 对每个特征的均值、方差维护指数移动平均；测试时使用这些运行估计。更新公式为 running_mean=momentum×running_mean+(1−momentum)×sample_mean，running_var 同理。这里采用移动平均的方式，原论文的测试流程略有不同。x 为 (N,D)，gamma、beta 为 (D,)；bn_param 中 mode 决定训练或测试，eps 用于数值稳定，momentum 控制运行估计的更新速度，running_mean、running_var 保存历史估计。返回 out（N,D）和反向传播所需的 cache。
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
        # [中文] 待完成：实现训练模式下的批归一化前向传播。用当前小批量计算均值和方差，按标准差（方差的平方根）归一化，再用 gamma、beta 缩放和平移，结果存入
        # out。反向传播需要的中间值存入 cache。结合 momentum 更新 running_mean、running_var。原论文链接见英文说明。
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        # [中文] 自行实现的代码区域结束。
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # [中文] 待完成：实现测试模式下的批归一化前向传播。用 running_mean、running_var 归一化输入，再用 gamma、beta 缩放和平移，将结果存入 out。
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # [中文] 自行实现的代码区域结束。
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    # [中文] 将更新后的运行均值存回 bn_param。
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

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
    批归一化的反向传播。建议先画计算图，再逐个中间节点应用链式法则。dout 为 (N,D)，cache 来自 batchnorm_forward；返回 dx（N,D）、dgamma（D,）和 dbeta（D,）。
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # [中文] 待完成：实现批归一化的反向传播，分别计算并保存 dx、dgamma、dbeta。可参考英文中给出的原论文。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward


    中文对照：
    批归一化反向传播的另一种实现。先在纸上推导并化简梯度表达式，更多提示见 notebook。输入输出与 batchnorm_backward 相同，使用相同的 cache，但化简后可能不需要其中所有值。
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
    # [中文] 待完成：实现批归一化反向传播的简化版本。结果存入 dx、dgamma、dbeta。原题提示：求出损失对中心化输入的梯度后，可进一步化简输入梯度的表达式；参考实现仅需一行。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

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
    层归一化的前向传播。对每个样本分别归一化，再用 gamma、beta 缩放和平移。训练与测试行为一致，不维护运行均值或方差。x 为 (N,D)，gamma、beta 为 (D,)，ln_param 中的 eps 用于数值稳定。返回 out（N,D）和反向传播所需的 cache。
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
    # [中文] 待完成：实现层归一化的前向传播，对输入归一化后用 gamma、beta 缩放和平移。原题提示：可以调整矩阵的组织方式，复用并略微修改训练模式下的批归一化实现。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

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
    层归一化的反向传播。可复用批归一化部分的推导。dout 为 (N,D)，cache 来自 layernorm_forward；返回 dx（N,D）、dgamma（D,）、dbeta（D,）。
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # [中文] 待完成：实现层归一化的反向传播。可以借鉴训练模式下的批归一化反向传播，前向传播部分关于矩阵变换的提示也适用。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

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

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.


    中文对照：
    inverted dropout 的前向传播。x 可为任意形状。dropout_param 中 p 是保留每个神经元输出的概率；mode='train' 时执行 Dropout，mode='test' 时直接返回输入。seed 用于让随机结果可复现，便于梯度检查。out 与 x 同形状，cache=(dropout_param,mask)，测试时 mask=None。注意实现 inverted dropout；这里的 p 表示保留概率，部分资料中则用 p 表示丢弃概率。
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
        # [中文] 待完成：实现训练模式下的 inverted dropout 前向传播，将使用的随机掩码存入 mask。
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        # [中文] 自行实现的代码区域结束。
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # [中文] 待完成：实现测试模式下 inverted dropout 的前向传播。
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # [中文] 自行实现的代码区域结束。

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.


    中文对照：
    inverted dropout 的反向传播。dout 是任意形状的上游梯度，cache 是前向传播保存的 (dropout_param,mask)。
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # [中文] 待完成：实现训练模式下 inverted dropout 的反向传播。
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        # [中文] 自行实现的代码区域结束。
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

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
    卷积层前向传播的朴素实现。x 为 (N,C,H,W)，w 为 (F,C,HH,WW)，每个卷积核覆盖全部 C 个输入通道；b 为 (F,)。conv_param 中 stride 是步幅，pad 是上下左右分别补零的宽度。不要直接修改原始 x。返回 out，形状为 (N,F,H',W')，其中 H'=1+(H+2×pad−HH)/stride，W'=1+(W+2×pad−WW)/stride；同时返回 cache=(x,w,b,conv_param)。
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # [中文] 待完成：实现卷积层的前向传播。原题提示：可用 np.pad 为输入补零。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b


    中文对照：
    卷积层反向传播的朴素实现。输入为上游梯度 dout 和前向传播保存的 cache=(x,w,b,conv_param)；返回损失对输入、权重、偏置的梯度 dx、dw、db。
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # [中文] 待完成：实现卷积的反向传播。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

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
    最大池化层前向传播的朴素实现。x 为 (N,C,H,W)；pool_param 指定 pool_height、pool_width 和 stride。本题不补零，并假设池化窗口能够整齐覆盖输入。输出为 (N,C,H',W')，其中 H'=1+(H−pool_height)/stride，W'=1+(W−pool_width)/stride；返回 out 和 cache=(x,pool_param)。
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # [中文] 待完成：实现最大池化的前向传播。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x


    中文对照：
    最大池化层反向传播的朴素实现。输入为上游梯度 dout 和前向传播保存的 cache=(x,pool_param)；返回损失对输入 x 的梯度 dx。
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # [中文] 待完成：实现最大池化的反向传播。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

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
    空间批归一化的前向传播。输入输出均为 (N,C,H,W)，gamma、beta 按通道缩放和平移，形状为 (C,)。bn_param 指定 mode、eps、momentum 及运行均值和方差。momentum=0 完全使用本轮统计量，momentum=1 不吸收新统计量，默认值为 0.9。返回 out 以及反向传播所需的 cache。
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # [中文] 待完成：实现空间批归一化的前向传播。原题提示：可调用上面实现的普通批归一化函数，参考实现不到五行。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)


    中文对照：
    空间批归一化的反向传播。dout 为 (N,C,H,W)，cache 来自前向传播；返回 dx（N,C,H,W）、dgamma（C,）、dbeta（C,）。
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # [中文] 待完成：实现空间批归一化的反向传播。原题提示：可复用上面实现的普通批归一化反向传播，参考实现不到五行。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

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
    空间组归一化的前向传播。将每个样本的 C 个通道分成 G 组，分别归一化，再按特征缩放和平移。x、out 为 (N,C,H,W)，gamma、beta 为 (1,C,1,1)；G 必须整除 C，gn_param 中 eps 用于数值稳定。返回 out 与 cache。
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
    # [中文] 待完成：实现空间组归一化的前向传播。原题提示：考虑如何重组矩阵，以复用层归一化或训练模式下批归一化的大部分逻辑。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)


    中文对照：
    空间组归一化的反向传播。dout 为 (N,C,H,W)，cache 来自前向传播；返回 dx（N,C,H,W），以及形状为 (1,C,1,1) 的 dgamma 和 dbeta。
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # [中文] 待完成：实现空间组归一化的反向传播，可参考层归一化的实现。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x


    中文对照：
    计算多类别 SVM 的损失和梯度。x 为 (N,C) 的类别评分数组，x[i,j] 是第 i 个样本对类别 j 的评分；y 为 (N,) 的标签数组，满足 0 <= y[i] < C。返回标量 loss 和损失对评分 x 的梯度 dx。
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # [中文] 待完成：复制你在作业一中完成的相应实现。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x


    中文对照：
    计算 Softmax 分类的损失和梯度。x 为 (N,C) 的类别评分数组，x[i,j] 是样本 i 对类别 j 的评分；y 为 (N,) 的标签数组，满足 0 <= y[i] < C。返回标量 loss 和损失对评分 x 的梯度 dx。
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # [中文] 待完成：复制你在作业一中完成的相应实现。
    N, C = x.shape

    x_ = x - np.max(x, axis=1, keepdims=True)
    x_ = np.exp(x_)
    probs = x_ / np.sum(x_, axis=1, keepdims=True)
    log_probs = np.log(probs)

    loss = -np.mean(log_probs[np.arange(N), y])

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
    return loss, dx
