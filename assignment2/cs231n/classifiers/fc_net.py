from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.


    中文对照：
    具有任意数量隐藏层的全连接神经网络，使用 ReLU 激活与 Softmax 损失，可选批归一化、层归一化和 Dropout。L 层网络的隐藏部分重复 L−1 次：affine → 可选归一化 → ReLU → 可选 Dropout；最后接 affine → Softmax。参数存入 self.params，由 Solver 训练。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.


        中文对照：
        初始化 FullyConnectedNet。hidden_dims 给出各隐藏层大小；input_dim、num_classes 分别是输入维度与类别数。dropout_keep_ratio 是保留概率，等于 1 时关闭 Dropout。normalization 可为 batchnorm、layernorm 或 None。reg 是 L2 正则化强度；weight_scale 是权重初始化标准差。dtype 决定计算精度：float32 较快，数值梯度检查建议使用 float64。若给定 seed，则 Dropout 使用固定随机种子，使梯度检查可以复现。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # [中文] 待完成：初始化网络参数并存入 self.params。第 l 层的权重和偏置分别使用 Wl、bl 这样的键名，例如 W1、b1、W2、b2。权重从均值为 0、标准差为
        # weight_scale 的正态分布采样，偏置置零。使用批归一化时，各层缩放和平移参数分别命名为 gamma1、beta1 等，gamma 初始化为 1，beta 初始化为
        # 0。
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        for i in range(1, self.num_layers+1):
            self.params[f"W{i}"] = weight_scale * np.random.randn(
                dims[i - 1], dims[i]
            )
            self.params[f"b{i}"] = np.zeros(dims[i])  
            if self.normalization in ("batchnorm", "layernorm") and i < self.num_layers:
                self.params[f"gamma{i}"] = np.ones(dims[i])
                self.params[f"beta{i}"] = np.zeros(dims[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # [中文] 每个 Dropout 层都需要 dropout_param 字典，从中读取保留概率和运行模式（训练或测试）。各层可以共用同一个 dropout_param。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # [中文] 批归一化需要记录均值和方差的运行估计，因此每层需接收各自的 bn_param。第一层前向传播使用 self.bn_params[0]，第二层使用
        # self.bn_params[1]，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        # [中文] 将全部参数转换为所需的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.


        中文对照：
        计算全连接网络的损失和梯度。X 的形状为 (N,d_1,...,d_k)，y 的形状为 (N,)。未提供 y 时返回测试模式下的类别评分 scores，形状为 (N,C)；提供 y 时返回训练模式下的标量损失 loss 和参数梯度字典 grads。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # [中文] 设置批归一化和 Dropout 的训练或测试模式，因为它们在这两种模式下的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # [中文] 待完成：实现全连接网络的前向传播，将 X 的各类别评分存入 scores。使用 Dropout 时，向每层传入 self.dropout_param；使用批归一化时，第 l
        # 个归一化层使用对应的 self.bn_params[l]。
        out = X
        caches = {}

        for i in range(1, self.num_layers):
            out, affine_cache = affine_forward(
                out, self.params[f"W{i}"], self.params[f"b{i}"]
            )

            bn_cache = None
            if self.normalization == "batchnorm":
                out, bn_cache = batchnorm_forward(
                    out,
                    self.params[f"gamma{i}"],
                    self.params[f"beta{i}"],
                    self.bn_params[i - 1],
                )
            elif self.normalization == "layernorm":
                out, bn_cache = layernorm_forward(
                    out,
                    self.params[f"gamma{i}"],
                    self.params[f"beta{i}"],
                    self.bn_params[i - 1],
                )
            out, relu_cache = relu_forward(out)
            caches[i] = (affine_cache, bn_cache, relu_cache)

            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                caches[i] = (affine_cache, bn_cache, relu_cache, dropout_cache)
                
        last = self.num_layers
        scores, last_cache = affine_forward(
            out, self.params[f"W{last}"], self.params[f"b{last}"]
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        # If test mode return early.
        # [中文] 测试模式下直接返回结果。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # [中文] 待完成：实现全连接网络的反向传播，分别将损失和梯度存入 loss、grads。使用 Softmax 数据损失，并加入带 0.5 系数的 L2 正则化；grads[k] 对应
        # self.params[k]。批归一化或层归一化的缩放参数 gamma、平移参数 beta 不需要正则化。
        loss, dout = softmax_loss(scores, y)

        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W * W)

        last = self.num_layers
        dout, dW, db = affine_backward(dout, last_cache)
        grads[f"W{last}"] = dW + self.reg * self.params[f"W{last}"]
        grads[f"b{last}"] = db

        for i in range(self.num_layers - 1, 0, -1):
            if self.use_dropout:
                affine_cache, bn_cache, relu_cache, dropout_cache = caches[i]
                dout = dropout_backward(dout, dropout_cache)
                dout = relu_backward(dout, relu_cache)
            else:
                affine_cache, bn_cache, relu_cache = caches[i]
                dout = relu_backward(dout, relu_cache)
                
            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            elif self.normalization == "layernorm":
                dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta

            dout, dW, db = affine_backward(dout, affine_cache)
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        return loss, grads
