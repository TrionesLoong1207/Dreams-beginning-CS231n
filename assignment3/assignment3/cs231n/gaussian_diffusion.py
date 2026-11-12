import torch
import torch.nn as nn
from tqdm.auto import tqdm
import math


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x_start",
        }, "objective must be either pred_noise (predict noise) or pred_x_start (predict image start)"

        # A helper function to register some constants as buffers to ensure that
        # they are on the same device as model parameters.
        # See https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # Each buffer can be accessed as `self.name`
        # [中文] 将一些常数登记为缓冲符的辅助函数, 以确保这些常数与模型参数位于同一个设备上。 见
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html 每个缓冲符都可以以
        # `self.name` 身份被访问
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        #############################################################################
        # Noise schedule beta and alpha values
        #############################################################################
        # [中文] 振动表 贝塔和阿尔法值
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        # [中文]  alpha_bar_t
        register_buffer("betas", betas)  # can be accessed as self.betas
        # [中文] 可用 self.betas 访问
        register_buffer("alphas", alphas)  # can be accessed as self.alphas
        # [中文] 可用 self.alphas 访问
        register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod
        # [中文]  self.alphas_cumprod

        #############################################################################
        # Other coefficients needed to transform between x_t, x_0, and noise
        # Note that according to Eq. (4) and its reparameterization in Eq. (14),
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # where noise is sampled from N(0, 1)
        #############################################################################
        # [中文] 变化所需的其他系数x_t,  x_0注:根据Eq.(4)及其在Eq中的再补偿。(14),从 N 中取样噪音(0, 1)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        # register_buffer(
        #     "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        # )
        # [中文]  register_buffer (" sqrt_recip_alphas_cumprod ",  torch.sqrt(1.0 /
        # alphas_cumprod) )  register_buffer ( " sqrt_recipm1_alphas_cumprod ",
        # torch.sqrt(1.0 / alphas_cumprod - 1)

        #############################################################################
        # For posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        #############################################################################
        # alpha_bar_{t-1}
        # [中文] 根据论文的 Eq. (6) 和 (7),对于后方 q(xt- 1 } x_t, x_0。 alpha_bar_ {t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        #################################################################
        # loss weight
        #################################################################
        # [中文] 损 重
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        return img * 2 - 1

    def unnormalize(self, img):
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t, t, noise):
        """Get x_start from x_t and noise according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            noise: (b, *) tensor. Noise from N(0, 1).
        Returns:
            x_start: (b, *) tensor. Starting image.


        中文对照：
        滚开x_start从x_t和噪音 根据Eq。(14)说明:x_t:  (b, *)高压。 吵闹的图像。 t:(b,)时间步骤 噪音:(b, *)来自N的噪音(0, 1)返回:x_start:  (b, *)压力 开始图像
        """
        x_start = None
        ####################################################################
        # TODO:
        # Transform x_t and noise to get x_start according to Eq.(4) and Eq.(14).
        # Look at the coeffs in `__init__` method and use the `extract` function.
        ####################################################################
        # [中文] 变换( T)x_t和噪音得到x_start根据Eq.(4)页:1(14)。看看在`__init__`方法及方法的使用`extract`函数。

        ####################################################################
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """Get noise from x_t and x_start according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            x_start: (b, *) tensor. Starting image.
        Returns:
            pred_noise: (b, *) tensor. Predicted noise.


        中文对照：
        获取噪音来自@ info: whatsthisx_t和x_start根据Eq.(14)说明:x_t:  (b, *)高压。 吵闹的图像。 t:(b,)时间步态。x_start:  (b, *)拖动。 开始图像。 返回:pred_noise:  (b, *)压力 预言噪音
        """
        pred_noise = None
        ####################################################################
        # TODO:
        # Transform x_t and noise to get x_start according to Eq.(4) and Eq.(14).
        # Look at the coeffs in `__init__` method and use the `extract` function.
        ####################################################################
        # [中文] 变换( T)x_t和噪音得到x_start根据Eq.(4)页:1(14)。看看在`__init__`方法及方法的使用`extract`函数。

        ####################################################################
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        """Get the posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        Args:
            x_start: (b, *) tensor. Predicted start image.
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
        Returns:
            posterior_mean: (b, *) tensor. Mean of the posterior.
            posterior_std: (b, *) tensor. Std of the posterior.


        中文对照：
        获取后端 q(xt-)1 } |  x_t,  x_0)根据Eq.(6)和(7)说明:x_start:  (b, *)压力,预言的开始图像。x_t:  (b, *)高压。 吵闹的图像。 t:(b,)拖拉。 时间步骤。 返回:posterior_mean:  (b, *)强力 后遗症的本性posterior_std:  (b, *)后院的石膏
        """
        posterior_mean = None
        posterior_std = None
        ####################################################################
        # We have already implemented this method for you.
        # [中文] 我们已经为你们采用了这种方法。
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        ####################################################################
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        """Sample from p(x_{t-1} | x_t) according to Eq. (6) of the paper. Used only during inference.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: int. Sampling time step.
            model_kwargs: additional arguments for the model.
        Returns:
            x_tm1: (b, *) tensor. Sampled image.


        中文对照：
        根据论文的 Eq. (6), p(x)Qt- 1 } x_t 样本。 仅在推断时使用。 参考: x_t: (b, *) 发声。 噪音图像。 t: int. 抽样时间步骤。 model_kwargs: 模型的额外参数。 返回: x_tm1:(b, *) 发音。 抽样图像。
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,)
        x_tm1 = None  # sample x_{t-1} from p(x_{t-1} | x_t)
        # [中文] 样本 xt - 1 } p( xt- 1 } x_t )

        ##################################################################
        # TODO: Implement the sampling step p(x_{t-1} | x_t) according to Eq. (6):
        #
        # - Steps:
        #   1. Get the model prediction by calling self.model with appropriate args.
        #   2. The model output can be either noise or x_start depending on self.objective.
        #      You can recover the other by calling self.predict_start_from_noise or
        #      self.predict_noise_from_start as needed.
        #   3. Clamp predicted x_start to the valid range [-1, 1]. This ensures the
        #      generation remains stable during denoising iterations.
        #   4. Get the mean and std for q(x_{t-1} | x_t, x_0) using self.q_posterior,
        #      and sample x_{t-1}.
        ##################################################################
        # [中文] TODO: 执行取样步骤 p(x)\\ t-1 } |  x_t)根据Eq.(6): -
        # 步骤:1。通过调用来获得模型预测self.model有适当的参数。2。模型输出可以是噪音,也可以是x_start取决于self.objective。您可以通过拨打self.predict_start_from_noise或self.predict_noise_from_start视需要而定。3。x_start至有效区域
        # [-1,  1这样做可确保代代相传期间保持稳定。4。得到平均值和标准为 q(x)t-1 } |  x_t,  x_0)
        # 使用self.q_posterior,以及样本 xt-1 }.
        
        ##################################################################

        return x_tm1

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def q_sample(self, x_start, t, noise):
        """Sample from q(x_t | x_0) according to Eq. (4) of the paper.

        Args:
            x_start: (b, *) tensor. Starting image.
            t: (b,) tensor. Time step.
            noise: (b, *) tensor. Noise from N(0, 1).
        Returns:
            x_t: (b, *) tensor. Noisy image.


        中文对照：
        根据论文Eq.(4)从q( x_t  x_0)样本。

        参数:x_start:  (b, *)启动图像。 t:(b,)时间步骤 噪音:(b, *)来自N的噪音(0, 1)返回:x_t:  (b, *)高压 吵闹的图像
        """

        x_t = None
        ####################################################################
        # TODO:
        # Implement sampling from q(x_t | x_0) according to Eq. (4) of the paper.
        # Hints: (1) Look at the `__init__` method to see precomputed coefficients.
        # (2) Use the `extract` function defined above to extract the coefficients
        # for the given time step `t`. (3) Recall that sampling from N(mu, sigma^2)
        # can be done as: x_t = mu + sigma * noise where noise is sampled from N(0, 1).
        # Approximately 3 lines of code.
        ####################################################################
        # [中文] TODO: 从q( 执行取样)x_t  |  x_0)根据Eq.(4)提示:(1)看着`__init__`用于查看预先计算系数的方法。 NAME OF
        # TRANSLATORS(2)使用`extract`函数,以提取给定时间步骤的系数`t` .  (3)回顾N(mu),Sigma的取样2(b)
        # 可采取下列行动:x_t= mu + sigma * 从 N 取样噪音的噪音(0, 1)大约3代码行。

        ####################################################################
        return x_t

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
        x_start = self.normalize(x_start)  # (b, *)
        noise = torch.randn_like(x_start)  # (b, *)
        target = noise if self.objective == "pred_noise" else x_start  # (b, *)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)
        loss = None

        ####################################################################
        # TODO:
        # Implement the loss function according to Eq. (14) of the paper.
        # First, sample x_t from q(x_t | x_0) using the `q_sample` function.
        # Then, get model predictions by calling self.model with appropriate args.
        # Finally, compute the weighted MSE loss.
        # Approximately 3-4 lines of code.
        ####################################################################
        # [中文] TODO: 根据论文的 Eq. (14) 执行损失函数。 首先, 使用 `q_sample` 函数从 q ( x_t ) < \\\\ > x_0 样本
        # x_t。 然后, 通过调用 self.model 和适当的参数来获得模型预测。 最后, 计算加权 MSE 损失。 大约 3 - 4 代码行。

        ####################################################################

        return loss


def extract(a, t, x_shape):
    """
    Extracts the appropriate coefficient values based on the given timesteps.

    This function gathers the values from the coefficient tensor `a` according to
    the given timesteps `t` and reshapes them to match the required shape such that
    it supports broadcasting with the tensor of given shape `x_shape`.

    Args:
        a (torch.Tensor): A tensor of shape (T,), containing coefficient values for all timesteps.
        t (torch.Tensor): A tensor of shape (b,), representing the timesteps for each sample in the batch.
        x_shape (tuple): The shape of the input image tensor, usually (b, c, h, w).

    Returns:
        torch.Tensor: A tensor of shape (b, 1, 1, 1), containing the extracted coefficient values
                      from a for corresponding timestep of each batch element, reshaped accordingly.


    中文对照：
    根据给定的时间步骤抽取适当的系数值。

    此函数根据给定的时间步骤 `t` 收集系数抗拉 `a` 的值,并重塑这些值,使其与要求的形状匹配,从而支持以给定形状 `x_shape` 的抗拉进行广播。

    参数: a ( torch.Tensor ): 形状 (T,) 的发光体, 包含所有时间步骤的系数值。 t ( torch.Tensor ): 形状 (b,) 的发热体, 代表每批样本的时间步骤。 x_shape (tuple): 输入图像发光体的形状, 通常是 (b, c, h, w)。

    返回: torch.Tensor:一个形状 (b, 1, 1, 1) 的柱状,包含每个批量元素相应时间步骤的抽取系数值,并相应调整。
    """
    b, *_ = t.shape  # Extract batch size from the timestep tensor
    # [中文] 从时间步骤强度中提取批量大小
    out = a.gather(-1, t)  # Gather the coefficient values from `a` based on `t`
    # [中文] 根据 `t` 从 `a` 收集系数值
    out = out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # Reshape to (b, 1, 1, 1) for broadcasting
    # [中文] 重塑为 (b, 1, 1, 1)用于广播
    return out


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper


    中文对照：
    线性排程, 在原始 ddpm 论文中提议
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ


    中文对照：
    https://openreview.net/forum?id=-NEXDKk8gZ 中提议的余弦计划
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training


    中文对照：
    在 https://arxiv.org/abs/2212.11972 - 图 8 中为图像 > 64x64 推荐的sigmoid 计划,如果在训练期间使用的话
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
