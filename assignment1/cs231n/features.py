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
    对 imgs 中每张图片应用 feature_fns 中的特征函数，再将同一张图片的各类特征拼接。imgs 为 (N,H,W,C)，第 i 个函数输出长度为 F_i 的向量。返回形状为 (N,F_1+...+F_k) 的数组，每一行对应一张图片；英文中将其写为每列，与给出的形状不一致。verbose 控制是否显示进度。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    # [中文] 用第一张图片确定特征维度。
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    # [中文] 确定特征维度后，一次分配用于存储全部特征的大数组；每一列保存一个样本的特征。
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    # [中文] 为其余图片提取特征。
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
    将 RGB 图像转换为灰度图。输入 rgb 是彩色图像，返回 gray 是灰度图像。
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
    为灰度或 RGB 图像提取方向梯度直方图（HOG）特征，返回特征向量 feat。实现改编自 skimage；参考文献为 Dalal、Triggs 的 CVPR 2005 论文，原链接保留在英文说明中。
    """

    # convert rgb to grayscale if needed
    # [中文] 必要时将 RGB 图像转为灰度图。
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    # [中文] 图像尺寸。
    orientations = 9  # number of gradient bins
    # [中文] 梯度方向直方图的区间数量。
    cx, cy = (8, 8)  # pixels per cell
    # [中文] 每个单元包含的像素数量。

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    # [中文] 计算水平方向的梯度。
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    # [中文] 计算竖直方向的梯度。
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    # [中文] 梯度幅值。
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation
    # [中文] 梯度方向。

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    # [中文] 水平方向的单元数量。
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    # [中文] 竖直方向的单元数量；计算梯度方向的积分图。
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        # [中文] 按当前方向范围筛选梯度，并建立该方向的积分图。
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # select magnitudes for those orientations
        # [中文] 选出这些方向对应的梯度幅值。
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
    基于色相提取颜色直方图。im 为 (H,W,C) 的 RGB 图像；nbin 为区间数，默认 10；xmin、xmax 默认 0、255；normalized 决定是否归一化。返回长度为 nbin 的一维特征向量。
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
# [中文] 以下是课程开发时尝试过的其他特征实现，不属于发给学生的代码。
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
    为灰度或 RGB 图像计算颜色直方图特征。im 是图像数组，nbin 是区间数，xmin、xmax 指定数值范围，normalized 决定是否归一化；返回特征 feat。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    # grayscale image
    # [中文] 灰度图像。
    if ndim == 2:
        imhist, bin_edges = np.histogram(im, bins=bins, density=normalized)
        return imhist
    # rgb image
    # [中文] RGB 彩色图像。
    elif ndim == 3:
        color_hist = np.array([])
        # loop through three color channels
        # [中文] 依次处理三个颜色通道。
        for k in range(3):
            # compute normalized histogram
            # [中文] 计算归一化后的直方图。
            imhist, bin_edges = np.histogram(im[:, :, k], bins=bins, density=normalized)
            imhist = imhist * np.diff(bin_edges)
            # concatenate histogram
            # [中文] 拼接各个直方图。
            color_hist = np.concatenate((color_hist, imhist))
        # return histogram
        return color_hist
    # unknown image type
    # [中文] 无法识别的图像类型。
    return np.array([])


def color_histogram_spatial(img, levels=3, nbin=4):
    """
    Color histogram over a pyramid.


    中文对照：
    基于图像金字塔的颜色直方图。
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
    RGB 三维联合颜色直方图，每个区间同时对应三个颜色通道。
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
# [中文] 原代码的删除区域结束标记。
