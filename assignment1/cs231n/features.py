from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.


    中文对照：
    考虑到图像的像素数据和可在单个图像上操作的若干特征,将所有特征应用到所有图像中,将每个图像的特征向量归为一体,并将所有图像的特征存储在一个矩阵中。

    输入: - imgs: N x H X W X C 像素数据数组用于 N 图像。feature_fns: k 特性函数列表。第 i 个特性函数应该将 H x W x D 数组作为输入并返回长度的(一维)数组F_i. - 动词: Boulean;如果是,则打印进度。

    返回:形状数组( N, F_1 +... + F_k ),其中每列是单个图像的所有特征的组合。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    # [中文] 使用第一张图像来确定特性尺寸
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    # [中文] 现在我们知道这些特征的维度, 我们可以分配一个大数组来存储所有的特性作为列。
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    # [中文] 提取其余图像的特征。
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image



    中文对照：
    将 RGB 图像转换为灰度

    参数:rgb: RGB 图像

    返回: 灰色: 灰度图像
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature



    中文对照：
    为图像计算梯度 (HOG) 特性的直方图

    从 skimage.feature.hog http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog 修改

    参考:用于人类检测的定向梯度直方图 Navneet Dalal 和 Bill Triggs, CVPR 2005

    参数: im: 输入灰度或 rgb 图像

    返回: 特征: 梯度 (HOG) 特性的直方图
    """

    # convert rgb to grayscale if needed
    # [中文] 需要时将 rgb 转换为灰度
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    # [中文] 图像大小
    orientations = 9  # number of gradient bins
    # [中文] 梯度数
    cx, cy = (8, 8)  # pixels per cell
    # [中文] 每个单元格像素

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    # [中文] 在 x 方向上计算梯度
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    # [中文] 在 Y 方向上计算梯度
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    # [中文] 梯度大小
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation
    # [中文] 梯度方向

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    # [中文] x 中的单元格数
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    # [中文] y 中计算方向的单元格数
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        # [中文] 创建此方向在此区域中分离方向的新集成图像
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # select magnitudes for those orientations
        # [中文] 选择这些方向的大小
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.


    中文对照：
    使用 hue 计算图像的颜色直方图。

    输入数: - im: H x W x C 像素数据数组用于 RGB 图像。 - nbin: 直方图包数。 (默认值: 10 ) - xmin: 最小像素值(默认值:0 ) - Xmax: 最大像素值 (默认值: 255) - 正常化: 是否对直方图进行常规化( 默认值: True)

    返回: nbin 长度的 1D 矢量,在输入图像的色度上显示直方图。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist


# ~~START DELETE~~
# These are some other features that we implemented to play around, but aren't
# distributing to students.
# [中文] 这些是我们用来玩耍的其他特征, 但不向学生分发。 校对:Portnoy
def color_histogram(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """Compute color histogram feature for an image

      Parameters:
        im : a numpy array of grayscale or rgb image
        nbin : number of histogram bins (default: 10)
        xmin : minimum pixel value (default: 0)
        xmax : maximum pixel value (deafult: 255)
        normalized : bool flag to normalize the histogram

      Returns:
        feat : color histogram feature



    中文对照：
    计算图像的颜色直方图特性

    参数: im: 一个灰度或 rgb 图像 nbin 的纽扣数组: 直方图箱数( 默认值: 10 ) xmin: 最小像素值( 默认数: 0 ) Xmax: 最大像素值 (deafult: 255 ) 普通化: 布尔标志使直方图正常化

    返回: 性能: 彩色直方图特性
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    # grayscale image
    # [中文] 灰度图像
    if ndim == 2:
        imhist, bin_edges = np.histogram(im, bins=bins, density=normalized)
        return imhist
    # rgb image
    # [中文] rgb 图像
    elif ndim == 3:
        color_hist = np.array([])
        # loop through three color channels
        # [中文] 通过三个颜色频道循环
        for k in range(3):
            # compute normalized histogram
            # [中文] 计算正态直方图
            imhist, bin_edges = np.histogram(im[:, :, k], bins=bins, density=normalized)
            imhist = imhist * np.diff(bin_edges)
            # concatenate histogram
            # [中文] 共形直方图
            color_hist = np.concatenate((color_hist, imhist))
        # return histogram
        return color_hist
    # unknown image type
    # [中文] 未知图像类型
    return np.array([])


def color_histogram_spatial(img, levels=3, nbin=4):
    """
    Color histogram over a pyramid.


    中文对照：
    金字塔上的彩色直方图
    """
    feats = []

    for level in range(1, levels + 1):
        chunks = np.array_split(img, level, axis=0)
        chunks = [np.array_split(chunk, level, axis=1) for chunk in chunks]
        for x in chunks:
            for chunk in x:
                feats.append(color_histogram_cross(chunk, nbin=nbin))

    return np.hstack(feats)


def color_histogram_cross(img, nbin=5, normalized=True):
    """
    RGB color histogram where our bins are 3 dimensional.


    中文对照：
    RGB 颜色直方图, 我们的文件夹是 3 维的。
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
# [中文] * 结束 * * * * 结束* * * * End Delete *
