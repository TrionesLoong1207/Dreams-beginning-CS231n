from __future__ import print_function
import numpy as np

try:
    from .im2col_cython import col2im_cython, im2col_cython
    from .im2col_cython import col2im_6d_cython
except ImportError:
    pass
    # print("""=========== You can safely ignore the message below if you are NOT working on ConvolutionalNetworks.ipynb ===========""")
    # print("\tYou will need to compile a Cython extension for a portion of this assignment.")
    # print("\tThe instructions to do this will be given in a section of the notebook below.")

from .im2col import *


def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.


    中文对照：
    以 im2col 和 col2im 为基础, 快速执行革命层的 前向传播。
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Check dimensions
    # [中文] 检查尺寸
    assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    # Create output
    # [中文] 创建输出
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Check dimensions
    # assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    # assert (H + 2 * pad - HH) % stride == 0, 'height does not work'
    # [中文] 检查维度维护 (W + 2 * pad -WW)% 斜度 = 0, “宽度不工作” 维护 (H + 2 * pap -HH)% 跳度 =0,“高度不工作
    # ”

    # Pad the input
    # [中文] 挂贴输入
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Figure out output dimensions
    # [中文] 找出产出层面
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    # [中文] 选择聪明的步调来执行 im2col 操作
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Now all our convolutions are a big matrix multiply
    # [中文] 现在,我们所有的卷积 是一个巨大的矩阵 乘以
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # Reshape the output
    # [中文] 重塑输出
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # Be nice and return a contiguous array
    # The old version of conv_forward_fast doesn't do this, so for a fair
    # comparison we won't either
    # [中文] 友善点, 返回一个毗连的数组 旧版本的conv_forward_fast 不这样做, 所以为了公平比较, 我们也不会这样做。
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


def conv_backward_im2col(dout, cache):
    """
    A fast implementation of the backward pass for a convolutional layer
    based on im2col and col2im.


    中文对照：
    以 im2col 和 col2im 为基础, 快速执行革命层的 反向传播。
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    dx = col2im_cython(
        dx_cols,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        filter_height,
        filter_width,
        pad,
        stride,
    )

    return dx, dw, db


conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the im2col method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the im2col method, which
    is not much faster than the naive method.


    中文对照：
    快速执行最大池化层的 前向传播。

    这选择了重塑方法与 im2col 方法。 如果集合区域是平方, 并且对输入图像进行拼贴, 那么我们就可以使用非常快速的重塑方法。 否则, 我们回到了 im2Col 方法, 这个方法不会比天真的方法快得多。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ("im2col", im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.


    中文对照：
    快速执行最大池化层的 反向传播。

    重塑方法与 im2col 方法之间的开关, 取决于使用何种方法生成缓存。
    """
    method, real_cache = cache
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)
    elif method == "im2col":
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
    """
    A fast implementation of the forward pass for the max pooling layer that uses
    some clever reshaping.

    This can only be used for square pooling regions that tile the input.


    中文对照：
    快速执行最大池化层的 前向传播, 使用一些聪明的重塑。

    这只能用于平整输入的平方集合区域。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    assert pool_height == pool_width == stride, "Invalid pool params"
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(
        N, C, H // pool_height, pool_height, W // pool_width, pool_width
    )
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    A fast implementation of the backward pass for the max pooling layer that
    uses some clever broadcasting and reshaping.

    This can only be used if the forward pass was computed using
    max_pool_forward_reshape.

    NOTE: If there are multiple argmaxes, this method will assign gradient to
    ALL argmax elements of the input rather than picking one. In this case the
    gradient will actually be incorrect. However this is unlikely to occur in
    practice, so it shouldn't matter much. One possible solution is to split the
    upstream gradient equally among all argmax elements; this should result in a
    valid subgradient. You can make this happen by uncommenting the line below;
    however this results in a significant performance penalty (about 40% slower)
    and is unlikely to matter in practice so we don't do it.


    中文对照：
    使用智能广播和重塑的最大池化层快速执行 反向传播。

    只有当 前向传播 使用 max_pool_forward_reshape 计算时,才能使用此选项。

    注意: 如果有多个参数, 此方法将会为输入的 All 参数指定梯度, 而不是选择一个参数。 在这种情况下, 梯度将实际上不正确。 但是这在实际中不太可能发生, 所以不应该有什么关系。 一个可能的解决方案是在所有参数中平均分割上游梯度; 这应该产生一个有效的子梯度。 您可以通过解开下面的线来做到这一点; 但是这会导致很大的性能处罚( 大约为 40% 慢一点), 并且在实践中也不太可能重要, 所以我们不这样做。
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = x_reshaped == out_newaxis
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


def max_pool_forward_im2col(x, pool_param):
    """
    An implementation of the forward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.


    中文对照：
    执行基于 im2col 的最大集合 前向传播。

    这并不比天真的版本快得多, 所以,如果可能,应该避免它。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    assert (H - pool_height) % stride == 0, "Invalid height"
    assert (W - pool_width) % stride == 0, "Invalid width"

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    An implementation of the backward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.


    中文对照：
    执行基于 im2col 的最大集合 反向传播。

    这并不比天真的版本快得多, 所以,如果可能,应该避免它。
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im_indices(
        dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
    )
    dx = dx.reshape(x.shape)

    return dx
