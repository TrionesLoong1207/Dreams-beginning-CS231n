"""Utility functions used for viewing and processing images.

中文对照：
用于查看和处理图像的实用功能。
"""

import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from imageio import imread
from PIL import Image



def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)


    中文对照：
    一个非常温和的图像模糊操作, 用作图像生成的常规化器。

    输入: - X:形状 (N, 3, H, W) 的图像数据

    返回: - X_blur: X 的模糊版本,形状 (N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.


    中文对照：
    预处理 shitchnet 的图像。

    减去像素平均值,除以标准差。
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8.

    中文对照：
    在图像上卸载预处理, 并转换为 Quint8。
    """
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.


    中文对照：
    从 URL 读取图像。 返回一个带有像素数据的缩略数组。 我们将图像写入一个临时文件, 然后读回它。 有点毛。
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, "wb") as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print("URL Error: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP Error: ", e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling


    中文对照：
    从磁盘装入图像并调整其大小。

    输入: - 文件名: 文件路径 - 大小: 调整后最小尺寸
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: the width, height values are currently flipped here, and we should
        # change the resampling method to BILINEAR to match the torch implementation
        # [中文] TODO: 宽度、 高度值目前在此翻转, 我们应该将重新采样方法更改为 BILINEAR 以匹配火炬执行
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
