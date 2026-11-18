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
        使用 SGD 训练线性分类器。X 为 (N,D)，y 为 (N,)，标签满足 0 <= y[i] < C。learning_rate 是学习率，reg 是正则化强度，num_iters 是更新次数，batch_size 是每次抽取的样本数，verbose 控制进度输出。返回每次迭代的损失列表。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        # [中文] 假设标签 y 取值为 0～K−1，其中 K 为类别数。
        if self.W is None:
            # lazily initialize W
            # [中文] 首次需要时再初始化 W。
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        # [中文] 用随机梯度下降优化 W。
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
            # [中文] 待完成：从训练集中抽取 batch_size 个样本及其对应标签，供本轮梯度下降使用。分别存入 X_batch 和 y_batch，形状应为 (batch_size,dim)
            # 和 (batch_size,)。原题提示：用 np.random.choice 生成索引；有放回抽样通常比无放回抽样更快。
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            # [中文] 计算损失和梯度。
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # [中文] 待完成：根据当前梯度和学习率更新权重。
            self.W -= grad * learning_rate

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
        用已训练的权重预测输入 X 的类别。X 为 (N,D)，返回长度为 N 的预测标签数组。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # [中文] 待完成：实现本方法，将预测标签存入 y_pred。
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
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
        计算小批量数据的损失和梯度，由具体分类器子类实现。X_batch 为 (N,D)，y_batch 为对应类别标签，reg 为正则化强度；返回标量损失及权重梯度。
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
      加载模型参数。
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
    使用多类别 SVM 损失的分类器子类。
    """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function

    中文对照：
    使用 Softmax 与交叉熵损失的分类器子类。
    """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
