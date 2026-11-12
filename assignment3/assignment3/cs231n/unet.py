import copy
from einops import rearrange
from torch import einsum

from torch import nn
import torch
import torch.nn.functional as F
import math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    """Upsample the image feature resolution a factor of 2.

    中文对照：
    将图像特性解析度的因数加起来2 .
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """Downsample the image feature resolution a factor of 2.

    中文对照：
    向下抽样图像特征分辨率为 2 系数。
    """
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    """RMSNorm layer which is compute-efficient simplified variant of LayerNorm.

    中文对照：
    正在计算图层Norm的简化变体的 RMSNorm 层。
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for time steps.

    中文对照：
    时间步的正弦位置嵌入。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """A conv block with feature modulation.

    中文对照：
    具有特质调控特征的 conv 区块。
    """

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # Scale and shift are used to modulate the output. This is a variant
        # of feature fusion, more powerful than simply adding the feature maps.
        # [中文] 缩放和移动用于调和输出。 这是特性聚合的变体, 比简单的添加特性映射更强大。
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """A ResNet-like block with context dependent feature modulation.

    中文对照：
    具有上下文特性调适的 ResNet 类块。
    """

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):

        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context)
            context = rearrange(context, "b c -> b c 1 1")
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        # Number of channels at each layer i.e. [d1, d2, ..., dn]
        # [中文] 每层 i.e 的频道数。 [d1, d2,..., dn]
        dims = [dim] + [dim * m for m in dim_mults]
        # Input and output for each U-Net block in downsampling layers
        # e.g. [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        # [中文] 下取样层中每个 U-Net 块的输入和输出e.g . [ (d1, d2),  (d2, d3),...,, (dn -),(dn) -
        #,1[原件:英文]
        in_out = list(zip(dims[:-1], dims[1:]))
        # Input and output for each U-Net block in upsampling layers
        # e.g. [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        # [中文] 每个U-Net区块在上层取样中的输入和输出e.g[(dn, dn-) .1(dn),(dn-), (d),(b) 和(d)1,dn - -,这是什么?2
        # ), ...,  (d2, d1) ]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # Encoding timestep as context
        # [中文] 将时间步骤编码为上下文
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Encoding condition (i.e. text embedding) as context
        # [中文] 编码条件 (i.e. 文本嵌入) 作为上下文
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Probability of dropping the condition during training
        # [中文] 训练期间降低条件的可能性
        self.uncond_prob = uncond_prob

        # UNet downsampling and upsampling blocks.
        # self.downs is a ModuleList of ModuleLists.
        # [中文] UNet 下标和上标区块。 self.downs 是模块列表。
        self.downs = nn.ModuleList([])
        # self.ups is a ModuleList of ModuleLists.
        # [中文]  self.ups 是模块列表。
        self.ups = nn.ModuleList([])

        ####################################################################
        # Downsampling blocks
        ####################################################################
        # [中文] 下游取样区块
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # TODO: Create one UNet downsampling layer `down_block` as a ModuleList.
            # It should be a ModuleList of 3 blocks [ResnetBlock, ResnetBlock, Downsample].
            # Each ResnetBlock operates on dim_in channels and outputs dim_in channels.
            # Make sure to pass the context_dim to each ResnetBlock.
            # The Downsample block operates on dim_in channels and outputs dim_out channels.
            # Make sure to exactly follow this structure of ModuleList in order to
            # load a pretrained checkpoint.
            ##################################################################
            # [中文] TODO: 创建一个UNet下游取样层`down_block`作为模块列表。 它应该是一个模块列表3区块 [ResnetBlock,
            # Resnetblock、 Downsample]。 每一个 ResneBlack
            # 都运行在dim_in渠道和产出dim_in频道。context_dim对每个 ResnetBlock。dim_in渠道和产出dim_out频道。
            # 请确保完全遵循模块列表的这个结构, 以便装载一个预先训练的检查站。

            ##################################################################
            self.downs.append(down_block)

        # Middle blocks
        # [中文] 中间区块
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        ####################################################################
        # Upsampling blocks
        ####################################################################
        # Create upsampling blocks by exactly mirroring the downsampling blocks.
        # self.ups will also be a ModuleList of ModuleLists.
        # Each BlockList will contain 3 blocks [Upsample, ResnetBlock, ResnetBlock].
        # [中文] 上标区块 创建上标区塊。 self.ups 也将是模块列表。 每个区块列表将包含 3 区块 [上标、 ResnetBlock、 ResnetBlock]
        # 。
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = None
            ##################################################################
            # TODO: Create one UNet upsampling layer as a ModuleList.
            # It should be a ModuleList of 3 blocks [Upsample, ResnetBlock, ResnetBlock].
            # This will mirror the corresponding downsampling block.
            # Don't forget to account for the skip connections by having 2 x dim_out
            # channels at the input of both ResnetBlocks.
            ##################################################################
            # [中文] TODO: 创建一个 Unit 上标层作为模块列表。 它应该是 3 区块[ 上标、 ResnetBlock、 Resneblock] 的模块列表。
            # 这将反映相应的下标区块。 不要忘了在 Resnetblock 的输入中设置 2 x dim_out 频道来计算跳过连接。

            self.ups.append(up_block)
            ##################################################################

        # Final convolution to map to the output channels
        # [中文] 映射到输出通道的最后演算
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """Classifier-free guidance forward pass. model_kwargs should contain `cfg_scale`.

        中文对照：
        无分类指导 前向传播。 model_kwargs 应包含 `cfg_scale`。
        """

        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        ##################################################################
        # TODO: Apply classifier-free guidance using Eq. (6) from
        # https://arxiv.org/pdf/2207.12598 i.e.
        # x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        #
        # You will have to call self.forward two times.
        # For unconditional sampling, pass None in`text_emb`.
        ##################################################################
        # [中文] TODO: 使用 https://arxiv.org/pdf/2207.12598 i.e 中的 Eq. (6) 应用无分类指导。 您需要两次调用
        # self.forward。 对于无条件的取样, 必须在 `text_emb` 中通过。

        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """Forward pass through the U-Net.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            time: Tensor of time steps of shape (batch_size,).
            model_kwargs: A dictionary of additional model inputs including
                "text_emb" (text embedding) of shape (batch_size, condition_dim).

        Returns:
            x: Output tensor of shape (batch_size, channels, height, width).


        中文对照：
         前向传播 通过 U-Net. 参数: x: 输入形状 (batch_size, channels, height, width). 时间: 形状 (batch_size,) 的时间级数。 model_kwargs: 包括形状 (batch_size, condition_dim) 的 " text_emb " (text embedding) 的额外模型输入字典。

        返回: x: 输出元件 (batch_size, channels, height, width) 的发光量。
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        # Embed time step
        # [中文] 嵌入时间步骤
        context = self.time_mlp(time)

        # Embed condition and add to context
        # [中文] 嵌入条件并添加到上下文
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
        if self.training:
            # Randomly drop condition
            # [中文] 随机下降状态
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # B x 1
            cond_emb = cond_emb * mask
        context = context + self.condition_mlp(cond_emb)

        # Initial convolution
        # [中文] 初始演进
        x = self.init_conv(x)

        ##################################################################
        # TODO: Process `x` through the U-Net conditioned on the context.
        #
        # 1. Downsampling:
        #    - Process `x` through each downsampling block with context.
        #    - After each ResNet block, save the output (feature maps) in a list or dict
        #      for use as skip connections in the upsampling path.
        #    - Make sure to pass the context to each ResNet block.
        #
        # 2. Middle:
        #    - Process `x` through the middle blocks with context.
        #
        # 3. Upsampling:
        #    - Process `x` through each upsampling block with context.
        #    - Before each ResNet block, concatenate the input with the corresponding
        #      skip connection from the downsampling path.
        #    - Make sure to pass the context to each ResNet block.
        ##################################################################
        # [中文] 待办事宜: 进程`x`通过U-Net 以上下文为条件。1. 下游取样: - 进程`x`- 在每个 ResNet 块之后, 保存输出(feature
        # maps)用于跳过更新路径中的连接的列表或剪贴。 - 请务必将上下文传送到每个 ResNet 块。2中间: - 进程`x`穿过中间街区 与上下文。3.
        # 取样: - 过程`x`- 在每个 ResNet 块之前,将输入与下标路径的相应跳过连接相连接。 ResNet 区块。

        ##################################################################

        # Final block
        # [中文] 最后一块
        x = self.final_conv(x)

        return x
