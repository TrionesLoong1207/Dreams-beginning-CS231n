from tensorflow.python.framework.ops import device_v2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].


    中文对照：
    计算文本和图像特征向量之间的对相相相近性。

    参数: text_features ( torch.Tensor ): 形状 (N, D)。 image_features ( torch.Tensor ) 形状: 形状 (M, D) 形状。

    返回时间:torch.Tensor:形状的相似矩阵(N, M),每个条目的位置(i, j)是相近的相近性, 而不是相近性的相近性和相似性。 text_features[i] 和image_features [j].
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################
    # [中文] TODO: 计算余弦相似性。 不要使用循环。

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # [中文] 代码结束

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.


    中文对照：
    使用 CLIP 模型执行零光图像分类。

    参数: clip_model ( torch.nn.Module ): 用于编码图像和文本的预先训练的 CLIP 模型。 clip_preprocess (Callable): 在编码前适用于每个图像的预处理功能。 图像 ( List[np.ndarray] ): 作为 NumPy 数组 (H x W x C) 的输入图像列表。 class_texts ( List[str] ): 用于零发分类的类标签字符串列表。 设备 ( torch.device ): 应进行计算的设备。 在将其传送到 clip_model 之前, 将 text_tokens 传送到设备。

    返回: List[str]:从给定 class_texts 中选择的每个图像的预测类标签。
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    # [中文] 查找图像的类标签。

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # [中文] 代码结束

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.


    中文对照：
    使用 CLIP 的简单图像检索系统。
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.


        中文对照：
        参数: clip_model ( torch.nn.Module ): 预训练的 CLIP 模型。 clip_preprocess (Callable): 预处理图像的函数。 图像 ( List[np.ndarray] ): 图像列表为 NumPy 数组 (H x W x C)。 设备 ( torch.device ): 该模型执行设备。
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        # [中文] TODO: 存储所有在检索方法中使用的必需对象变量。 请注意, 您应该在此同时处理所有图像, 并避免对每个文本查询进行重复计算。
        # 您可能最终不会使用上述相似功能来进行最优化的计算。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        pass
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.


        中文对照：
        获取最接近输入文字的上方k 图像的索引。 您可能会发现 torch.Tensor.topk 方法有用。

        参数: 查询 (str): 文本查询。 k (int): 返回上方的 k 图像。

        返回: List[int]:最相似的上方-k 图像的指数。
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        # [中文] 获取上K图像的索引。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).


    中文对照：
    生成 RGB 图像上方的彩色分割覆盖。

    参数:segmentation_mask  ( np.ndarray: 2D 形状数组(H, W),加上类指数。np.ndarray3D 形状数组(H, W, 3)RGB 图像。 alpha(float): 重叠的透明度系数(美元)0=仅图像,1= 仅面罩)。

    返回: np.ndarray:图像带有部分重叠(形状: (H, W, 3), dtype: viint8)。
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    # [中文] 使用固定色图生成每个类的确定颜色
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        # [中文] 用于确定论
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    # [中文] 为分隔面罩创建颜色图像
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    # [中文] 与原始图像混合
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU).

    中文对照：
    计算Union (IoU) 上的中间线平均值。
    """
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.


        中文对照：
        初始化 DINOS 部分模型。

        这定义了一个简单的神经网络,旨在将 DINO 特征向量分类为分割类别。 它包括模型初始化、优化和损失函数设置。

        参数: 设备 ( torch.device ): 在 (CPU or CUDA) 上运行模型的设备。 num_classes (int): 分区类数。 inp_dim (int, optional):输入 DINO 特性的尺寸。
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        # [中文] TODO: 定义一个非常轻的发光模型、 优化和丢失特征, 以将每个 DINO 特征向量分类为 seg. 类。 它可以是线性层或两层神经网络。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.


        中文对照：
        利用所提供的训练数据训练分解模型。

        参数: X_train ( torch.Tensor ): 形状 (N, D) 的输入特征向量。 Y_train ( torch.Tensor ):形状 (N,) 的地面真实标签。 num_iters (int, optional): 优化步骤的数目。
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        # [中文] 训练你的模型`num_iters`台阶。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.


        中文对照：
        对给定的测试 DINO 特征向量进行推断。

        参数: X_test ( torch.Tensor ): 形状 (N, D) 的输入特征向量。

        返回:形状(N,)torch.Tensor:预测类指数。
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        # [中文] 训练你的模型`num_iters`台阶。

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 代码结束
        return pred_classes