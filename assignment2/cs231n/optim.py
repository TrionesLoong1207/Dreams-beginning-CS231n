import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.


    中文对照：
    执行普通随机梯度下降。

    配置格式: - learning_rate: 标量 学习率。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.


    中文对照：
    以动量进行悬浮梯度下降。

    配置格式: - learning_rate: 标量 学习速度。 - 动量: 0 和 1 之间有 scalar 表示动量值。 设置动量 = 0 降低到 sgd。 -速度: 与用于存储梯度梯度平均值的 w 和 dw 相同形状的NumPy 数组。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # [中文] TODO: 执行动量更新公式。 在 next_w 变量中存储更新值。 您也应该使用并更新速度 v。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.


    中文对照：
    使用 RMSProp 更新规则,该规则使用平方梯度移动平均值来设定适应性的每参数学习率。

    配置格式: - learning_rate: 标量 学习速度。 - decay_rate: 0 和 1 之间的卡拉尔 给出正方形梯度缓存的衰减速度。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # [中文] TODO: 执行 RMSprop 更新公式, 在 next_w 变量中存储下一个 w 值。 不要忘记更新存储在 config['cache'] 中的缓存值
    # 。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.


    中文对照：
    使用 Adam 更新 规则, 其中包括 梯度 及其平方 的移动平均值 和 偏差 校正 术语。

    配置格式: - learning_rate: 标量 学习率。 - 贝塔1: 移动梯度一阶矩平均值的衰减率。- 贝塔2: 移动梯度二阶矩平均值的衰变率。 -- epsilon: 平滑时使用的较小的标量,以避免除以零。 - m: 移动梯度平均值。 - v: 平方梯度的移动平均数。 - t: 迭代号。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # [中文] TODO: 执行 Adam 更新公式, 在 next_w 变量中存储 W 的下一个值。 不要忘记更新在配置中存储的 m、 v 和 t 变量。 注意:
    # 为了匹配引用输出, 请在任何计算中使用 t 之前修改。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 代码结束

    return next_w, config
