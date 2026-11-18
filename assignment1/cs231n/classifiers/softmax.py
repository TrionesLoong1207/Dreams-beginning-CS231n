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
    Softmax 损失的朴素实现（使用循环）。每批有 N 个样本，特征维度为 D，类别数为 C。W 的形状为 (D,C)，X 为 (N,D)，y 为 (N,)；y[i]=c 表示样本 i 的类别为 c，且 0 <= c < C。reg 是正则化强度。返回损失标量及与 W 同形状的权重梯度。
    """
    # Initialize the loss and gradient to zero.
    # [中文] 将损失和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    # [中文] 计算损失及其梯度。
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        # [中文] 以数值稳定的方式计算概率。
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        # [中文] 归一化。
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        # [中文] 损失为正确类别概率的负对数。
        for j in range(10):
            if j == y[i]:
                grad = -1 + p[j]
            else:
                grad = p[j]
            dW[:,j] += grad * X[i]

    # normalized hinge loss plus regularization
    # [中文] 归一化后的合页损失加上正则化项。
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # [中文] 待完成：计算损失对权重的梯度并存入 dW。可以在计算损失的循环中同时累积梯度，因此可能需要调整上面已有的计算代码。


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.


    中文对照：
    Softmax 损失的向量化实现。输入和输出与 softmax_loss_naive 相同。
    """
    # Initialize the loss and gradient to zero.
    # [中文] 将损失和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # [中文] 待完成：以向量化方式计算 Softmax 损失，存入 loss。
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    ps = np.exp(scores)
    ps /= np.sum(ps, axis=1, keepdims=True)
    logps = np.log(ps)
    loss = -np.mean(logps[np.arange(num_train),y])
    loss += reg * np.sum(W*W)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # [中文] 待完成：实现 Softmax 损失梯度的向量化计算，将结果存入 dW。原题提示：复用计算损失时已有的中间结果，通常比重新从头计算更方便。
    dscores = ps.copy()
    dscores[np.arange(num_train), y] -= 1

    dW = X.T.dot(dscores)
    dW = dW / num_train + reg * 2 * W
    

    return loss, dW
