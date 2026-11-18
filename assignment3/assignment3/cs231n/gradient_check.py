import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at


    中文对照：
    计算标量函数 f 在数组 x 各个分量上的数值梯度。f 只接收一个数组参数，x 是求梯度的位置。
    """

    fx = f(x)  # evaluate function value at original point
    # [中文] 计算原始位置的函数值。
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    # [中文] 依次遍历 x 的各个索引。
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        # [中文] 计算 x+h 处的函数值。
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # [中文] 增加步长 h。
        fxph = f(x)  # evalute f(x + h)
        # [中文] 计算 f(x+h)。
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        # [中文] 计算 f(x−h)。
        x[ix] = oldval  # restore
        # [中文] 恢复先前保存的值。

        # compute the partial derivative with centered formula
        # [中文] 用中心差分公式计算偏导数。
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        # [中文] 斜率。
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension
        # [中文] 继续检查下一个维度。

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.


    中文对照：
    对接收 NumPy 数组并返回 NumPy 数组的函数计算数值梯度。
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
    计算使用输入、输出 blob 的函数的数值梯度。函数 f 接收若干输入 blob 和一个输出 blob，检查时对输入逐项扰动，并结合输出端梯度计算输入梯度。具体 blob 接口约定见英文原说明。
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
    随机抽取几个参数维度，只返回这些维度的数值梯度。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # [中文] 增加步长 h。
        fxph = f(x)  # evaluate f(x + h)
        # [中文] 计算 f(x+h)。
        x[ix] = oldval - h  # increment by h
        # [中文] 增加步长 h。
        fxmh = f(x)  # evaluate f(x - h)
        # [中文] 计算 f(x−h)。
        x[ix] = oldval  # reset
        # [中文] 重置状态。

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
