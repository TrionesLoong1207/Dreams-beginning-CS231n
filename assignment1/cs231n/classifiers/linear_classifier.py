from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.


        中文对照：
        使用随机梯度下降来训练此线性分类器。

        输入: - X: 形状的NumPy 数组(N, D)包含训练数据; 每个维度都有 N 训练样本。 - y: 形状的NumPy 数组(N,)含有训练标签;y[i]=c 意味着X[i]有标签0c < C 类。learning_rate:  (float)优化学习率。 - 区域:(float)- 正则化的力量。num_iters:  (integer)优化时要采取的步骤数量batch_size:  (integer)每一步使用的训练实例数量。 - 动词:(boolean)如果正确, 在优化时打印进度。

        产出:列出每次训练迭代的损失函数值。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        # [中文] 假设 y 值为 0 ...K - 1, K 是类数
        if self.W is None:
            # lazily initialize W
            # [中文] 稍稍初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        # [中文] 运行随机梯度下降以优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # [中文] 待办事宜: 样本batch_size用于这一轮梯度下降的训练数据及其相应标签中的元素。X_batch和相应的标签y_batch;
            # 取样后X_batch应该有形状(batch_size, dim)和y_batch应该有形状(batch_size,)提示:
            # 使用np.random.choice来生成指数。用替换进行取样比不替换取样要快。


            # evaluate loss and gradient
            # [中文] 评估损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # [中文] 执行参数更新 TODO: 使用梯度和学习率更新加权数。


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.


        中文对照：
        使用该线性分类器经过训练的权重来预测数据点的标签。

        输入: - X:包含训练数据的形状 (N, D) 的NumPy 数组;每个维度D都有N训练样本。

        返回时间: -y_pred: X 中数据的预测标签。y_pred是一个1- 长度为N的维数组,每个元素为整数,给出了预测的类别。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # [中文] TODO: 应用此方法。 将预测的标签保存在y_pred 中。

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W


        中文对照：
        计算损失函数及其衍生物。 子类将超过此值。

        投入: -X_batch:一个形形的NumPy 数组(N, D)含有一小批N数据点;每个点都有维度D。 -y_batch:一个形形的NumPy 数组(N,)包含小批的标签。 - ưμ㼯A(float)正则化的力量。

        返回: 含有: - 以单浮标损失 - 相对于self.W 的梯度; 与 W 形状相同的数组
        """
        pass

    def save(self, fname):
      """Save model parameters.

      中文对照：
      保存模型参数。
      """
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = {"W": self.W}
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters.

      中文对照：
      装入模型参数。
      """
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.W = params["W"]
        print(fname, "loaded.")
        return True


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function

    中文对照：
    使用多级 SVM 损失函数的子类
    """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function

    中文对照：
    使用 Softmax + Cross-entropy 损失函数的子类
    """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
