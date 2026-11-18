from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.


    中文对照：
    三层卷积网络：卷积 → ReLU → 2×2 最大池化 → 全连接 → ReLU → 全连接 → Softmax。输入图像批量采用 (N,C,H,W) 格式。
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.


        中文对照：
        初始化卷积网络。input_dim 为 (C,H,W)，num_filters 和 filter_size 为卷积核数量与边长，hidden_dim 为全连接隐藏层大小，num_classes 为输出类别数。weight_scale 是随机初始化标准差，reg 是 L2 正则化强度，dtype 指定数值类型。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # [中文] 待完成：初始化三层卷积网络的参数。权重从均值为 0、标准差为 weight_scale 的高斯分布采样，偏置置零，全部存入 self.params。卷积层用
        # W1、b1，隐藏全连接层用 W2、b2，输出全连接层用 W3、b3。本题卷积层的补零和步幅设置会保持输入的空间高度与宽度，具体设置见 loss() 开头。
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * (H//2) * (W//2), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.


        中文对照：
        计算三层卷积网络的损失和梯度。输入输出遵循 fc_net.py 中 TwoLayerNet 的相同接口。
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        # [中文] 将 conv_param 传入卷积前向传播；这里的补零与步幅设置用于保持输入的空间尺寸。
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        # [中文] 将 pool_param 传给最大池化的前向传播。
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # [中文] 待完成：实现三层卷积网络的前向传播，将各类别评分存入 scores。可调用已导入的 cs231n/fast_layers.py 和 cs231n/layer_utils.py
        # 中的函数。
        out, conv_relu1_pool_cache =  conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, fc1_relu2_cache = affine_relu_forward(out, W2, b2)
        scores, fc2_cache = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # [中文] 待完成：实现三层卷积网络的反向传播，损失存入 loss，参数梯度存入 grads。数据损失使用 Softmax，并加入 L2 正则化；grads[k] 对应
        # self.params[k]。为匹配参考实现，L2 正则化项应包含 0.5 系数。
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        dout, dW3, db3 = affine_backward(dscores, fc2_cache)
        dout, dW2, db2 = affine_relu_backward(dout, fc1_relu2_cache)
        dX, dW1, db1 = conv_relu_pool_backward(dout, conv_relu1_pool_cache)

        grads["W1"], grads["b1"] = dW1 + self.reg * W1, db1
        grads["W2"], grads["b2"] = dW2 + self.reg * W2, db2
        grads["W3"], grads["b3"] = dW3 + self.reg * W3, db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。

        return loss, grads
