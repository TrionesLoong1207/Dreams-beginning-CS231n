import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverPytorch(object):
    """
    A CaptioningSolverPytorch encapsulates all the logic necessary for
    training Pytorch based image captioning models.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, the instance variable solver.loss_history
    will contain a list of all losses encountered during training.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolverPytorch works on a model object that must conform to the following
    API:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.


    中文对照：
    描述SolverPytorch 包涵了训练基于 Pytoch 的图像字幕模型所需的全部逻辑。

    要训练模型,您将首先构建一个描述求解实例, 将模型、 数据集和各种选项 (learning rate, batch size, etc) 传递给构建者。 然后您将调用火车 () 方法来运行优化程序并训练模型。

    列车 () 方法返回后, 实例变量 solver.loss_history 将包含训练中遇到的所有损失列表。

    示例用法可能看起来是这样的:

    数据 =load_coco_data() 型号 = MyAwesome 模型 ()hidden_dim = 100求解器 = 描述求解( 模型, 数据, )optim_config ={  'learning_rate':  1e-3, },  num_epochs = 10,  batch_size = 100,  print_every = 100 )  solver.train()

     描述SolverPytorch 用于一个必须符合下列 API 的模型对象:

    输入: - 特征: 提供图像、 形状( N、 D - 说明: 图像描述的排列, 形状 (N, T), 其中每个元素都位于范围 ( 0, V)。

    返回: - 损失: 标量 给出损失 - 梯度: 与self.params绘图参数名称为这些参数的损失梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:

        - learning_rate: Learning rate of optimizer.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.


        中文对照：
        构造一个新的描述求解实例。

        所需参数: - 模型: 符合上述 API 的模型对象 - 数据: load_coco_data 的训练和验证数据词典

        可选论点:

        - learning_rate:优化的学习率。 - batch_size:用于计算训练期间损失和梯度的微型插头的大小。 -num_epochs:训练期间运行的训练轮次数。 - print_every:整数;训练损失将打印到每个print_every迭代。 - 动词:布尔恩;如果设定为错误,训练期间将不打印输出。
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        # [中文] 卸载关键词参数
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(list(model.params.values()), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        # [中文] 如果存在额外关键字参数, 丢弃错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

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
        self.loss_history = []


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.


        中文对照：
        做一个单一的梯度更新。 这是由列车 () 命名的, 不应该手工命名。
        """
        # Make a minibatch of training data
        # [中文] 制作少量的训练数据
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions = torch.from_numpy(captions).long()
        features = torch.from_numpy(features)
        loss = self.model.loss(features, captions)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.loss_history.append(loss.detach().numpy())

    def train(self):
        """
        Run optimization to train the model.


        中文对照：
        运行优化来训练模型。
        """
        for k, v in self.model.params.items():
          v.requires_grad_()

        num_train = self.data["train_captions"].shape[0]
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

            # At the end of every epoch, increment the epoch counter.
            # [中文] 在每一个训练轮次的尽头, 递增训练轮次的柜台。
            epoch_end = (t + 1) % iterations_per_epoch == 0

        for k, v in self.model.params.items():
          v.requires_grad_(False)

