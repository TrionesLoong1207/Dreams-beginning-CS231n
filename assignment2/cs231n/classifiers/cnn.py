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
    具有以下结构的三层革命网络:

    -  - 2x2  - relu -

    该网络在由N图像组成的 (N, C, H, W) 形状的小型数据包上运作,每个图像的高度为H和宽度为W,并有C输入通道。
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
        初始化一个新的网络。

        输入: - input_dim: Tuple (C, H, W) 提供输入数据大小 - num_filters: 用于卷积层的滤波器数量 - filter_size: 在卷积层使用滤波器的宽度/高度 - hidden_dim:在全连接的隐藏层中使用的单位数量 - num_classes: 从最后折叠层生成的分数数。 - weight_scale: 为权重随机初始化提供标准差的弧度。 - 重: 标量 提供 L2 正则化强度 - dtype: 用于计算的非浮数据类型。
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
        # [中文] TODO: 初始化三层革命网络的权重和偏差。 权重应该从高斯中心开始。0.0标准差等于weight_scale;
        # 偏差应初始化为零。所有权重和偏差都应存储在字典中self.params。 使用键盘存储电流层的权重和偏差'W1'和'b1';
        # 使用键'W2'和'b2'对于隐藏的方形层和键的权重和偏差'W3'和'b3'对输出平面的权重和偏差。 IMPORT: 对于此任务,
        # 您可以假设第一个卷积层的倾斜和曲折。 选择图层以便 ** 保存输入的宽度和高度**。查看损失()函数的起始点,看看是如何发生的。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.


        中文对照：
        评估三层革命网络的损失和梯度。

        输入/ 输出: API 与 TwoLayerNet 在 fc_net.py 中相同。
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        # [中文] 通过 conv_param 到 前向传播,用于为保存输入空间大小而选择的电流层平面和斜面
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        # [中文] 通过 pool_param 到最大共享层的 前向传播
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
        # [中文] TODO: 执行前向传播对于三层变换网,计算 X 的类分并将其存储到分数变量中。记住,您可以使用 cs231n/
        # 定义的函数。fast_layers.py和Cs231n/layer_utils.py执行中的(already imported) .

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

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
        # [中文] TODO:
        # 执行反向传播对于三层演变网,将损失和梯度储存在损失和分级变量中。使用软max计算数据损失,并确保grads[k]保持梯度self.params[k].
        # 不要忘了添加 L2 正则化! 注意: 要保证您的执行与我们相符, 您通过自动测试, 请确保您的 L2正则化包含一个因素:0.5以简化梯度的表达式。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        return loss, grads
