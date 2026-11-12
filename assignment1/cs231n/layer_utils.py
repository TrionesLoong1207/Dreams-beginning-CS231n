from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass


    中文对照：
    方便度层, 绕行折叠变形, 之后是 RELU。 @ info: whatsthis

    输入数: - x: 输入角层- w, b: 角层的权重

    返回一个图普: - 退出: 从 RELU - 缓存中输出: 要给给 反向传播 的对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer


    中文对照：
     松脂便利层 反向传播
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

