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
        kNN 的训练阶段只保存数据。X 为 (num_train,D) 的训练样本数组，y 为一维标签数组；y[i] 是 X[i] 的类别。
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
        预测测试数据的标签。X 为 (num_test,D)，k 为参与投票的最近邻数量；num_loops 选择距离计算的实现。返回长度为 num_test 的预测标签数组。
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
        对测试样本和训练样本分别使用一层循环，计算 X 中每个测试样本与 self.X_train 中每个训练样本的欧氏距离。输入 X 为 (num_test,D)，输出 dists 为 (num_test,num_train)，dists[i,j] 对应第 i 个测试样本与第 j 个训练样本。
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
                # [中文] 待完成：计算第 i 个测试样本与第 j 个训练样本的 L2 距离，写入 dists[i,j]。不允许再对特征维度使用循环，也不允许使用 np.linalg.norm()。
                diff = X[i] - self.X_train[j]
                dist = np.sqrt(np.sum(diff**2))
                dists[i,j] = dist
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops


        中文对照：
        仅对测试样本使用一层循环，计算其到全部训练样本的距离。输入输出与 compute_distances_two_loops 相同。
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
            # [中文] 待完成：一次计算第 i 个测试样本与全部训练样本的 L2 距离，写入 dists[i,:]。不允许使用 np.linalg.norm()。
            diff = X[i] - self.X_train
            dists[i,:] = np.sqrt(np.sum(diff**2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops


        中文对照：
        不使用显式循环，一次计算全部测试样本到全部训练样本的距离。输入输出与 compute_distances_two_loops 相同。
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
        # [中文] 待完成：不使用任何显式循环，计算所有测试样本与所有训练样本的 L2 距离，结果存入 dists。只使用基本数组运算，不调用 scipy 中的距离函数或
        # np.linalg.norm()。原题提示：将 L2 距离展开为矩阵乘法和两个可通过广播相加的求和项。
        test_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train**2, axis=1)
        cross = X @ self.X_train.T
        
        dist_sq = test_sq + train_sq - cross*2
        dists = np.sqrt(dist_sq)
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
        根据距离矩阵预测类别。dists 为 (num_test,num_train)，每行保存一个测试样本到全部训练样本的距离；k 指定参与投票的最近邻数量。返回每个测试样本的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # [中文] 长度为 k 的列表，保存第 i 个测试样本的 k 个最近邻标签。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # [中文] 待完成：根据距离矩阵，找出第 i 个测试样本距离最近的 k 个训练样本，再从 self.y_train 取出它们的标签，存入 closest_y。原题提示：查阅
            # numpy.argsort。
            dist_idx = np.argsort(dists[i])
            k_idx = dist_idx[:k]
            for j in k_idx:
                closest_y.append(self.y_train[j])

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # [中文] 待完成：统计 closest_y 中哪个标签出现次数最多，将该标签存入 y_pred[i]。如果多个标签票数并列，选择编号较小的标签。
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
        return y_pred
