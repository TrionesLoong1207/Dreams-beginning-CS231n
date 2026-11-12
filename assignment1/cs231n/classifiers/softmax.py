from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W


    中文对照：
    软体损失功能, 幼稚执行 (with loops)

    输入有D维度,有C级,我们用N小插头操作。

    输入: - W: 包含权重的形状 (D, C) 的缩略数组。 - X: 含有少量数据的缩略图 (N, D) 的缩微数组。 - y: 包含训练标签的缩略图(N,) 缩略图; y[i] = c 表示 X[i] 有标签 c, 其中 0 \ > c < C. - reg: (float) 正则化强度

    返回一个图普: - 以单浮标损失 - 相对于权重的梯度; 与 W 形状相同的数组
    """
    # Initialize the loss and gradient to zero.
    # [中文] 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    # [中文] 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        # [中文] 以数字稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        # [中文] 正常化
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        # [中文] 负日志概率为损失


    # normalized hinge loss plus regularization
    # [中文] 正常的关节损失加上正则化
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # [中文] TODO: 计算损失函数的梯度并将其存储为 dW。 相反, 首先计算损失然后计算衍生物, 在计算损失的同时计算衍生物可能比较简单。 因此,
    # 您可能需要修改上面的一些代码来计算梯度。


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.


    中文对照：
    软体损失函数, 矢量化版本。

    投入和产出与softmax_loss_naive相同。
    """
    # Initialize the loss and gradient to zero.
    # [中文] 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # [中文] TODO: 执行软体损失的矢量化版本, 存储导致损失的结果。


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # [中文] TODO: 执行一个向量化的梯度版本, 以存储结果为 dW. Hint: 与其从零开始计算梯度, 不如重新使用一些您用来计算损失的中间值比较容易。


    return loss, dW
