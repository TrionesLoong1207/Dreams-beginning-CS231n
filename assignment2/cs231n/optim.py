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
    普通随机梯度下降（SGD）。config 中的 learning_rate 是标量学习率。
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
    带动量的 SGD。config 中 learning_rate 为学习率，momentum 为 0 到 1 之间的动量系数，取 0 时退化为普通 SGD；velocity 与权重同形状，用于保存更新中的历史累积量。
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
    # [中文] 待完成：实现带动量的参数更新，将更新后的权重存入 next_w，同时更新速度变量 v。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。
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
    RMSProp：利用梯度平方的移动平均，为不同参数调整更新步长。config 中 learning_rate 为学习率，decay_rate 控制统计量衰减，epsilon 用于避免除零，cache 保存梯度平方的移动平均。
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
    # [中文] 待完成：实现 RMSProp 更新，将新权重存入 next_w，并更新 config['cache'] 中保存的梯度平方统计量。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

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
    Adam：结合梯度及梯度平方的指数移动平均，并进行偏差修正。config 中 learning_rate 为学习率；beta1、beta2 分别控制一阶矩与二阶矩估计的衰减；epsilon 保证数值稳定；m、v 保存两类移动平均，t 是迭代次数。
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
    # [中文] 待完成：实现 Adam 更新，将新权重存入 next_w，并更新 config 中的 m、v、t。为匹配参考输出，必须先更新 t，再在后续计算中使用它。
    config["t"] += 1
    
    config["m"] = config["beta1"] * config["m"] + (1-config["beta1"]) * dw
    m_hat = config["m"] / (1 - config["beta1"] ** config["t"])

    config["v"] = config["beta2"] * config["v"] + (1-config["beta2"]) * dw ** 2
    v_hat = config["v"] / (1 - config["beta2"] ** config["t"])

    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # [中文] 自行实现的代码区域结束。

    return next_w, config
