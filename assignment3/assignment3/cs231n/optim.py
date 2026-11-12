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


中文对照：
此文件执行各种用于训练神经网络的一阶更新规则。 每个更新规则都接受当前权重和相对于这些权重的损失梯度, 并产生下一组权重。 每一个更新规则都有相同的界面:

def 更新( w, dw, 配置=无):

输入: - w: 给定当前加权数的缩放数组。 - dw: 与给定 w. - 配置: 字典包含超参数值, 如学习率、 动量等。 如果更新规则要求对许多迭代的缓存值, 那么配置也会持有这些缓存值。

返回: - next_w: 更新后的下一个点。 - 配置: 配置词典将传送到更新规则的下一个迭代。

注:对于大多数更新规则来说,默认学习率可能效果不佳;但是,其他超参数的默认值对于各种不同的问题应当效果良好。

为了提高效率,更新规则可以进行现场更新、变异和设置next_w等于w。
"""

import numpy as np


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
    v *= config["momentum"]
    v -= config["learning_rate"] * dw
    w += v
    next_w = w
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
    rho = config["decay_rate"]
    lr = config["learning_rate"]
    eps = config["epsilon"]
    config["cache"] *= rho
    config["cache"] += (1.0 - rho) * dw ** 2
    step = -(lr * dw) / (np.sqrt(config["cache"]) + eps)
    w += step
    next_w = w

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
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]
    t, m, v = config["t"], config["m"], config["v"]
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw * dw)
    t += 1
    alpha = config["learning_rate"] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    w -= alpha * (m / (np.sqrt(v) + eps))
    config["t"] = t
    config["m"] = m
    config["v"] = v
    next_w = w

    return next_w, config
