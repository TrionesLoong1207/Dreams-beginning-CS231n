from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.


    中文对照：
    溶剂包罗了训练分类模型所需的所有逻辑。 溶剂使用optim.py 中定义的不同更新规则进行随机梯度下降。

    求解器接受训练和有效数据及标签,以便能定期检查训练和验证数据的分类准确性,以警惕是否过配。

    要训练模型,您将首先构建一个解答器实例, 将模型、 数据集和各种选项 (learning rate, batch size, etc) 传递给构造函数。 然后您将调用火车( ) 方法来运行优化程序并训练模型。

    火车法返回后,model.params将包含在训练过程中对认证集进行最佳表现的参数。此外,实例变量solver.loss_history将包含训练过程中遇到的所有损失和实例变量列表solver.train_acc_history和solver.val_acc_history将列出每个训练轮次的训练和验证模式的准确率。

    示例用法可能看起来是这样的:

    数据 = {'X_train':# 训练数据'y_train':# 训练标签'X_val':# 验证数据'y_val'# 验证标签} 模型 = MyAwesome Model ()hidden_size = 100, 区域=10求解器 = 解答器( 模型, 数据, )update_rule = 'sgd',  optim_config ={  'learning_rate':  1e-4, },  lr_decay = 0.95,  num_epochs = 5,  batch_size = 200,  print_every = 100 )  solver.train()

     解答器在必须符合下列 API 的模型对象上工作:

    - model.params 必须是一个字典映射字符串参数名称, 以包含参数值的缩放数组。

    - model.loss(X, y)必须是一个计算训练时间损失和梯度以及测试时间分类分数的函数,输入和产出如下:

    输入数: - X: 提供形状( N, d_1,..., d_k ) - y: 标签的矩阵, 形状 (N,) 的标签, X 的标签, y[i] 是 X[i] 的标记。

    返回:如果 y 是无,请运行一个测试时间 前向传播 并返回: - 分数: 形状 (N, C) 的矩阵给 X 提供分类分数, 在 scores[i, c] 给 X[i] 的 c 类分数。

    如果 y 不是 0, 则在前方运行训练时间。 @ info: whatsthis反向传播并返回一个图普: - 损失: 标量 提供损失 - 毕业: 与self.params绘图参数名称为这些参数的损失梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.


        中文对照：
        构造一个新的解析器实例。

        所需参数: - 模型:符合上述API的模型对象 -- -- 数据:包含以下内容的训练和验证数据词典:'X_train': 数组, 形状 (A)N_train,  d_1, ...,  d_k) 训练图像'X_val': 数组, 形状 (A)N_val,  d_1, ...,  d_k) 验证图像'y_train': 数组, 形状(N_train,)用于训练图像的标签'y_val': 数组, 形状(N_val,)用于验证图像的标签

        任择论点: -update_rule: 显示更新规则名称的字符串optim.py默认值'sgd' . -  optim_config:包含超参数的字典将传送到选定的更新规则。每个更新规则都需要不同的超参数(见optim.py)但所有更新规则都要求'learning_rate'参数,所以该参数应该始终存在。 - lr_decay:学习率的浮点数衰减;在每一个训练轮次之后,学习率乘以这个数值。 -batch_size:用于计算训练期间损失和梯度的微型蓄水池的大小。num_epochs:训练期间要运行的训练轮次数。 - print_every:整数;训练损失 将每页打印print_every迭代。 - 动词: Boulean; 如果设置为假的, 训练期间不会打印输出。num_train_samples:用于检查训练准确性的训练样本数量;默认为1000; 设置为无,使用整个训练集。 -num_val_samples:用于检查 val 精确度的验证样本数量;默认为无,该样本使用整个验证集。 -checkpoint_name:如果不是无,那么每个训练轮次都保存示范检查站。
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        # [中文] 卸载关键词参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        # [中文] 如果存在额外关键字参数, 丢弃错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        # [中文] 确保更新规则存在, 然后用实际函数替换字符串名称
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.


        中文对照：
        设置一些用于优化的簿记变量。 不要手工调用它。
        """
        # Set up some variables for book-keeping
        # [中文] 为簿记设置一些变量
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        # [中文] 为每个参数制作一个 optim_config 的深度副本
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.


        中文对照：
        做一个单一的梯度更新。 这是由列车 () 命名的, 不应该手工命名。
        """
        # Make a minibatch of training data
        # [中文] 制作少量的训练数据
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        # [中文] 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        # [中文] 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.


        中文对照：
        检查所提供数据的模型的准确性。

        输入: - X: 数据、形状( N, d_1,..., d_k ) - y: 标签、形状 (N,) - num_samples 的数组: 如果不是无, 则在 num_samples 数据点上分样数据, 并且只测试模型。 - batch_size: 将X 和 y 拆分成这样大小的批次, 以避免使用过多的内存。

        返回: - acc: 标量 给出按模型正确分类的事例的分数。
        """

        # Maybe subsample the data
        # [中文] 也许分样的数据
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        # [中文] 按批次计算预测
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.


        中文对照：
        运行优化来训练模型。
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            # [中文] 也许是印刷训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            # [中文] 在每个训练轮次的末期, 不断递增的训练轮次反面 和衰败的学习率。
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            # [中文] 检查列车和 val 精确度 在第一个迭代, 最后一次迭代, 在每一个训练轮次的末端。
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model
                # [中文] 跟踪最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        # [中文] 训练结束时,把最好的参数换成模型
        self.model.params = self.best_params
