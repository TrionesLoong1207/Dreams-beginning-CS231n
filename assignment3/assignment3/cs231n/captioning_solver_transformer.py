import numpy as np

from . import optim
from .coco_utils import sample_coco_minibatch, decode_captions

import torch


class CaptioningSolverTransformer(object):
    """
    A CaptioningSolverTransformer encapsulates all the logic necessary for
    training Transformer based image captioning models.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, the instance variable solver.loss_history
    will contain a list of all losses encountered during training.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeTransformerModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolverTransformer works on a model object that must conform to the following
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
    管理基于 Transformer 的图像描述模型训练。构造时传入模型、数据和训练配置，调用 train() 开始训练；接口约定与示例保留在英文说明中。
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
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
        构造图像描述训练器。必需参数为 model 和 load_coco_data 返回的 data；learning_rate、batch_size、num_epochs、print_every 分别控制学习率、批大小、训练轮数、日志间隔，verbose 控制日志输出。
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        # [中文] 解包关键字参数。
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        # [中文] 如果传入了多余的关键字参数，则报错。
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

        self.idx_to_word = idx_to_word

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.


        中文对照：
        初始化优化过程所需的记录变量；不应手动调用该方法。
        """
        # Set up some variables for book-keeping
        # [中文] 初始化用于记录训练过程的变量。
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.


        中文对照：
        执行一次梯度更新；该方法由 train() 调用，不应手动调用。
        """
        # Make a minibatch of training data
        # [中文] 构造一个训练用的小批量。
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self.model._null

        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)

        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Run optimization to train the model.


        中文对照：
        运行优化过程以训练模型。
        """
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            # [中文] 按设置决定是否打印训练损失。
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter.
            # [中文] 每个 epoch 结束时，将 epoch 计数加一。
            epoch_end = (t + 1) % iterations_per_epoch == 0

    def transformer_temporal_softmax_loss(self, x, y, mask):
        """
        A temporal version of softmax loss for use in RNNs. We assume that we are
        making predictions over a vocabulary of size V for each timestep of a
        timeseries of length T, over a minibatch of size N. The input x gives scores
        for all vocabulary elements at all timesteps, and y gives the indices of the
        ground-truth element at each timestep. We use a cross-entropy loss at each
        timestep, summing the loss over all timesteps and averaging across the
        minibatch.

        As an additional complication, we may want to ignore the model output at some
        timesteps, since sequences of different length may have been combined into a
        minibatch and padded with NULL tokens. The optional mask argument tells us
        which elements should contribute to the loss.

        Inputs:
        - x: Input scores, of shape (N, T, V)
        - y: Ground-truth indices, of shape (N, T) where each element is in the range
             0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
          the scores at x[i, t] should contribute to the loss.

        Returns a tuple of:
        - loss: Scalar giving loss


        中文对照：
        RNN 使用的时间序列 Softmax 损失。x（N,T,V）给出每个时间步的词表评分，y（N,T）给出真实词索引。在各时间步计算交叉熵、求和，再按批大小 N 平均。mask（N,T）指定哪些位置计入损失，从而忽略用于补齐序列的 NULL 标记。返回标量 loss。
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
