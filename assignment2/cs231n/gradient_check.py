from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at


    中文对照：
    一个天真地执行 f at x - f 的数值梯度应该是一个函数, 需要一个参数 - x 是点(numpy array)以评价梯度
    """

    fx = f(x)  # evaluate function value at original point
    # [中文] 在原始点评价函数值
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    # [中文] x 中所有索引的循环
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        # [中文] 在 x+h 处评价函数
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # [中文] 递增数x h
        fxph = f(x)  # evalute f(x + h)
        # [中文] evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        # [中文] 评估 f(x- h)
        x[ix] = oldval  # restore
        # [中文] 恢复

        # compute the partial derivative with centered formula
        # [中文] 用中心公式计算部分衍生物
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        # [中文] 斜坡
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension
        # [中文] 步进到下一个维度

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.


    中文对照：
    评估一个函数的数值梯度,该函数将接受一个网状数组,并返回一个网状数组。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a
    blob where outputs will be written. For example, f might be called like:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size


    中文对照：
    计算在输入和输出浮标上运行的函数的数字梯度。

    我们假设 f 接受数个输入块作为参数, 然后再接受一个输出的块。 例如, f 可以称为:

    f (x, w, out)

    输入 blobs 的 x 和 w, F 的结果将被写入。

    输入: - f: 函数 - 输入: 输入块的图普 - 输出: 输出块 - h: 步数大小
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.


    中文对照：
    试样一些随机元素, 仅返回此维的数值。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # [中文] 递增数x h
        fxph = f(x)  # evaluate f(x + h)
        # [中文] 评估 f(x + h)
        x[ix] = oldval - h  # increment by h
        # [中文] 递增数x h
        fxmh = f(x)  # evaluate f(x - h)
        # [中文] 评估 f(x- h)
        x[ix] = oldval  # reset
        # [中文] 重设

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
