"""Utility functions used for viewing and processing images.

中文对照：
用于查看和处理图像的工具函数。
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
    对图像施加轻微模糊，在图像生成中作为一种正则化。输入与输出的形状均为 (N,3,H,W)。
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
    为 SqueezeNet 预处理图片：按通道减去像素均值，再除以标准差。
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8.

    中文对照：
    撤销图片预处理，再转回 uint8。
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
    从 URL 读取图片，并返回像素 NumPy 数组。当前实现先下载到临时文件，再从文件读取。
    """
    try:
        f = urllib.request.urlopen(url)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
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
    从磁盘读取并缩放图片。filename 是文件路径，size 指定缩放后较短边的长度。
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # TODO: the width, height values are currently flipped here, and we should
        # change the resampling method to BILINEAR to match the torch implementation
        # [中文] 原代码的待修正说明：此处宽、高参数的顺序颠倒；重采样应改用 BILINEAR，以与 PyTorch 实现一致。
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))
    return img
