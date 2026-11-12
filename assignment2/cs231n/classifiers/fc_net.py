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
    多层全连接的神经网络的类。

    网络包含任意数的隐藏层, ReLU 非线性, 以及软负损失函数。 这还将执行 Dropout（随机失活） 和批量/ 层的归一化选项。 对于有 L 层的网络来说, 结构将会是

    {ffine - [批量/层规范] - relu - [Dropout（随机失活）]} x (L - 1) - 折叠 - 软式

    中,批量/级正态和 Dropout（随机失活） 是可选的, {...} 块是重复的 L - 1 倍。

    可学习参数存储在self.params 字典中,并将使用解析器类学习。
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
        初始化一个新的全连网。

        投入: -hidden_dims:一个显示每个隐藏层大小的整数列表。 -input_dim:一个以输入大小表示的整数。 - num_classes:一个整数,显示要分类的类别数量。 -dropout_keep_ratio: 浮点数，范围在0和1给Dropout（随机失活）强度。如果dropout_keep_ratio = 1那么网络就不应该使用Dropout（随机失活）- 归一化:网络应该使用哪种类型的归一化。有效的数值是“batchnorm”、“trainnorms”或“not for no regular”(the default)- 说明:标量给L2正则化力量。 - 证明。 - 标量 give L2 strange structure.weight_scale: 标量 给出加权数随机初始化的标准差。 - dtype: a numpy data type 对象; 将使用此数据类型进行所有计算。 浮点32 速度较快, 但准确度较低, 因此您应该使用浮点64 进行数字梯度检查。 - 种子: 如果不是无, 那么将随机种子传送到 Dropout（随机失活） 层。 这将使得 Dropout（随机失活） 层具有确定性, 这样我们就可以通过梯度检查模型。
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
        # [中文] TODO: 初始化网络参数, 在self.params字典。
        # W1和b1中第一层的储存权重和偏差;第二层使用W2和b2等。0标准差等于weight_scale。当对伽马1 和 贝塔1
        # 的第一个层使用批量归一化、储存比例和转移参数时,应初始化为零;对于第二层使用伽马2 和 比塔2 等。比例参数应初始性为1, 移动参数应初步初始化成零。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # [中文] 当使用 Dropout（随机失活） 时,我们需要向每个 Dropout（随机失活） 层传递一个 dropout_param 字典,以使该层知道
        # Dropout（随机失活） 概率和模式(火车/测试)。 您可以向每个 Dropout（随机失活） 层传递相同的dropout_param。
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
        # [中文] 批量归一化 我们需要追踪运行手段和差异 所以我们需要通过一个特殊的bn_param对象对应于每个批次解析层。您应该通过self.bn_params [
        # 0致《公约》第4条的下列内容:
        # 与《公约》、《公约》第四条第1款、第四条第2款、第3款、以及第3款和第3款的相对重要性;与《京都议定书》第3条第1款有关的事项。前向传播在第一批归一化层中,self.bn_params
        # [ 1致《公约》第4条的下列内容:
        # 与《公约》、《公约》第四条第1款、第四条第2款、第3款、以及第3款和第3款的相对重要性;与《京都议定书》第3条第1款有关的事项。前向传播第二批归一化层等
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        # [中文] 将所有参数投放到正确的数据类型。
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
        为全连接的网计算损失和梯度。

        输入数: - X: 形状输入数据的数组( N, N)d_1, ...,  d_ky: 标签、 形状的数组(N,) .  y[i]给定标签X[i] .

        返回:如果 y 是 0,则运行测试时间前向传播和返回: - 分数: 形状数组(N, C)给定分类分数, 在哪里scores[i, c]是分类的分数X[i]和C类。

        如果 y 不是 0, 则在前方运行训练时间。反向传播并返回一个图普: - 损失: 计算损失的斜体值 - 梯度: 与self.params,绘制参数名称为这些参数的损失梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # [中文] 设置批量规范参数和 Dropout（随机失活） 参数的火车/测试模式,因为它们在训练和测试期间的行为不同。
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
        # [中文] TODO: 对全连接的网安装 前向传播, 计算 X 的类分并将其存储在分数变量中。 当使用 Dropout（随机失活） 时, 您需要将
        # self.dropout_param 传送到每个 Dropout（随机失活） 前向传播。 当使用批量归一化时, 您将需要将批量归一化的
        # self.bn_params [ 0 ] 传送到第一组交配层的 前向传播, 通过 self.bn_params [ 1] 传送到第二批次交配层 的
        # 前向传播 等。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        # If test mode return early.
        # [中文] 如果测试模式提前返回。
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
        # [中文] TODO: 执行反向传播对于完全连接的网。在 grads
        # 字典中存储损失变量和梯度中的损失。使用软max计算数据损失,并确保grads[k]保持梯度self.params[k]。不要忘记添加 L2 归一化。
        # 当使用批量/ 层归一化时, 您不需要对比例和转换参数进行归一化。 注意: 要确保您的执行与我们的一致, 您通过自动测试, 请确保您的 L2正则化包含一个因素
        #:0.5以简化梯度的表达式。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束

        return loss, grads
