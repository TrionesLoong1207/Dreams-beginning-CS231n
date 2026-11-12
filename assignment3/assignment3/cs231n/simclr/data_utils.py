from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms


    中文对照：
    此函数返回单个训练图像中数据增强的构成。 完成以下行。 提示: 查看 torchvision.transforms 中的可用函数
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    # [中文] 应用颜色抖动的亮度= 0.4 、对比度= 0.4 、饱和度=0.4 和hue= 0.1 的转换
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Check out transformation functions defined in torchvision.transforms #
        # The first operation is filled out for you as an example.
        ##############################################################################
        # Step 1: Randomly resize and crop to 32x32.
        # [中文] TODO: 启动您的代码。 提示: 检查 torchvision.transforms 中定义的转换函数。 第一个操作被填充为示例。 步骤 1:
        # 随机调整大小和裁剪为 32x32。
        transforms.RandomResizedCrop(32),
        # Step 2: Horizontally flip the image with probability 0.5
        # [中文] 步骤 2: 水平翻转图像, 概率为 0.5

        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        # [中文] 步数 3: 概率为 0.8 时, 应用颜色杂音( 您可以使用上面定义的 " color_jitter " )。

        # Step 4: With a probability of 0.2, convert the image to grayscale
        # [中文] 步骤 4: 概率为 0.2 时, 将图像转换为灰度

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # [中文] 代码结束
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.


    中文对照：
    CIFAR10 数据集。
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO: Start of your code.                                                  #
            #                                                                            #
            # Apply self.transform to the image to produce x_i and x_j in the paper #
            ##############################################################################
            # [中文] TODO: 启动您的代码。 对图像应用 self.transform 来生成论文中的 x_i 和 x_j
            pass
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            # [中文] 代码结束

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target