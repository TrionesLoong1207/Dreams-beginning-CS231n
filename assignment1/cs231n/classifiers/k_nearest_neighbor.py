from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance

    中文对照：
    使用 L2 距离的 kNN 分类器。
    """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].


        中文对照：
        训练分类器。对于 k 近邻分类器，训练阶段只需保存训练数据。

        输入：
        - X：形状为 (num_train, D) 的 NumPy 数组，包含 num_train 个训练样本，每个样本有 D 个维度。
        - y：形状为 (N,) 的 NumPy 数组，包含训练标签；y[i] 是 X[i] 的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].


        中文对照：
        使用此分类器预测测试数据的标签。

        输入：
        - X：形状为 (num_test, D) 的 NumPy 数组，包含 num_test 个测试样本，每个样本有 D 个维度。
        - k：参与投票以确定预测标签的最近邻数量。
        - num_loops：决定使用哪一种实现来计算训练点与测试点之间的距离。

        返回值：
        - y：形状为 (num_test,) 的 NumPy 数组；y[i] 是测试点 X[i] 的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.


        中文对照：
        计算 X 中每个测试点与 self.X_train 中每个训练点之间的距离。使用双重嵌套循环，分别遍历测试数据与训练数据。

        输入：X 是形状为 (num_test, D) 的 NumPy 数组，包含测试数据。

        返回值：dists 是形状为 (num_test, num_train) 的 NumPy 数组，其中 dists[i, j] 是第 i 个测试点与第 j 个训练点之间的欧氏距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # [中文] 待完成：计算第 i 个测试点与第 j 个训练点之间的 L2 距离，并将结果存入 dists[i, j]。不允许再对特征维度使用循环，也不允许使用
                # np.linalg.norm()。
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops


        中文对照：
        计算 X 中每个测试点与 self.X_train 中每个训练点之间的距离，仅使用一个循环遍历测试数据。

        输入与输出：与 compute_distances_two_loops 相同。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # [中文] 待完成：计算第 i 个测试点与所有训练点之间的 L2 距离，将结果存入 dists[i,:]。不允许使用 np.linalg.norm()。
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops


        中文对照：
        计算 X 中每个测试点与 self.X_train 中每个训练点之间的距离，不使用显式循环。

        输入与输出：与 compute_distances_two_loops 相同。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # [中文] TODO: 计算所有测试点和所有训练点之间的 l2 距离,而不使用任何明确的循环,然后将结果存储为 dists。 您应该只使用基本的数组操作来执行此函数;
        # 特别是, 您不应该使用来自 scip 的函数, 也不应该使用 np.linalg.norm()。 HINT: 尝试使用矩阵乘法和两个广播总和来构造 l2
        # 的距离。

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].


        中文对照：
        根据测试点与训练点之间的距离矩阵,预测每个测试点的标签。

        输入: - dists: 形状 (num_test, num_train) 的缩略数组,其中 dists[i, j] 给出了 第 i 个 测试点和 第 j 个 训练点的距离。

        返回: - y:一个包含测试数据预测标签的形状(num_test,)的纽扣数组,其中y[i]是测试点X[i]的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # [中文] 保存于第 i 个 测试点附近的 k 标签的长度 k 列表。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # [中文] TODO: 使用距离矩阵查找 第 i 个 测试点最近的 k 邻居, 并使用 self.y_train 查找这些邻居的标签。 将这些标签保存在
            # closest_y 中。 提示: 查找函数 numpy.argsort。


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # [中文] TODO: 既然您已经找到了 k 最近的邻居的标签, 您就需要在标签列表 closest_y 中找到最常见的标签。 将标签保存在 y_pred [i] 中。
            # 通过选择小标签来断开链接。


        return y_pred
