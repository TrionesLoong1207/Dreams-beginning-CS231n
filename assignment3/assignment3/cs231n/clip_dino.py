import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
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
    计算文本特征与图像特征之间两两余弦相似度。text_features 为 (N,D)，image_features 为 (M,D)，返回 (N,M) 的相似度矩阵，每行对应一条文本，每列对应一张图像。
    """
    similarity = None
    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################
    # [中文] 待完成：计算余弦相似度，不允许使用 for 循环。
    text_norm = torch.linalg.norm(text_features, dim=1, keepdim=True)
    image_norm = torch.linalg.norm(image_features, dim=1, keepdim=True)

    text_normalized = text_features / text_norm
    image_normalized = image_features / image_norm

    similarity = text_normalized @ image_normalized.T
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # [中文] 自行实现的代码区域结束。

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
    用 CLIP 执行零样本图像分类。通过预训练模型编码图片与候选类别文本，根据相似度确定类别；模型、预处理、图片、类别名称和设备等参数的具体形式见英文列表。
    """
    
    pred_classes = []

    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################
    # [中文] 待完成：预测图片的类别标签。
    text_tokens = clip.tokenize(class_texts).to(device)
    image_batch = []
    for image in images:
        image_batch.append(clip_preprocess(Image.fromarray(image)))
    
    image_batch = torch.stack(image_batch).to(device)
                           
    text_features = clip_model.encode_text(text_tokens)
    image_features = clip_model.encode_image(image_batch)

    similarity = get_similarity_no_loop(text_features, image_features)
    pred_indices = similarity.argmax(dim=0)
    for i in pred_indices.tolist():
        pred_classes.append(class_texts[i])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # [中文] 自行实现的代码区域结束。

    return pred_classes
  

class CLIPImageRetriever:
    """
    A simple image retrieval system using CLIP.


    中文对照：
    基于 CLIP 的简单图像检索系统。
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
        初始化图像检索器。clip_model 为预训练 CLIP，clip_preprocess 为图片预处理函数，images 是若干 (H,W,C) 的 NumPy 图片数组，device 指定运行设备。
        """
        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################
        # [中文] 待完成：保存 retrieve 方法需要的对象属性。应在此处一次性处理全部图片，避免每次文本查询都重复编码。计算量更优的实现未必需要调用上面定义的相似度函数。
        self.clip_model = clip_model
        self.device = device

        image_batch = torch.stack([
            clip_preprocess(Image.fromarray(image))
            for image in images
        ]).to(device)

        self.image_features = clip_model.encode_image(image_batch)
        self.image_features = self.image_features / torch.linalg.norm(
            self.image_features, dim=1, keepdim=True
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
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
        根据 query 文本检索最相似的前 k 张图片，返回它们在图片列表中的索引。原题提示：可参考 torch.Tensor.topk。
        """
        top_indices = []
        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################
        # [中文] 待完成：返回相关度最高的前 k 张图片的索引。
        text_tokens = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        similarity = get_similarity_no_loop(text_features, self.image_features)

        _, indices = similarity.topk(k, dim=1)
        top_indices = indices[0].tolist()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return top_indices

  
class DavisDataset:
    def __init__(self):
        import tensorflow_datasets as tfds
        self.tfds = tfds
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(self.tfds.as_numpy(self.davis))
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
    将类别索引分割掩码叠加到 RGB 原图上，以不同颜色表示各类别。掩码为 (H,W)，图片为 (H,W,3)，混合强度等参数见英文约定；返回彩色叠加图。
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    # [中文] 使用固定色表，为每个类别生成可复现的颜色。
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        # [中文] 保证结果可复现。
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    # [中文] 将分割掩码转换成彩色图以便显示。
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image
    # [中文] 与原图混合显示。
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """Compute the mean Intersection over Union (IoU).

    中文对照：
    计算平均交并比（mean IoU）。
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
        初始化轻量 DINO 分割分类模型及其优化器和损失函数，用于将图像块的 DINO 特征分类为分割类别。
        """

        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################
        # [中文] 待完成：定义轻量 PyTorch 分类模型、优化器和损失函数，将每个 DINO 特征向量映射为分割类别。可以使用一个线性层或双层神经网络。
        self.device = device
        self.num_classes = num_classes
        self.inp_dim = inp_dim

        self.model = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )
        self.criterion = nn.CrossEntropyLoss()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        pass

    def train(self, X_train, Y_train, num_iters=500):
        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.


        中文对照：
        用给定 DINO 特征与对应类别标签训练分割模型。X_train 为 (N,D) 的特征张量，其余训练选项见英文列表。
        """
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        # [中文] 待完成：训练模型 num_iters 次迭代。
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)

        self.model.train()
        
        for iter in range(num_iters):
            scores = self.model(X_train)
            loss = self.criterion(scores, Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        pass
    
    @torch.no_grad()
    def inference(self, X_test):
        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.


        中文对照：
        对测试 DINO 特征 X_test（N,D）进行推理，返回对应的分割类别预测。
        """
        pred_classes = None
        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################
        # [中文] 待完成：训练模型 num_iters 次迭代。
        self.model.eval()
        X_test = X_test.to(self.device)
        
        scores = self.model(X_test)
        pred_classes = torch.argmax(scores, dim=1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # [中文] 自行实现的代码区域结束。
        return pred_classes
