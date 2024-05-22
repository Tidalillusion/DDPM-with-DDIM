
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified time steps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    ARGS：
    V ： the tensor that sqrt or 1 - sqrt； and the sqrt is the line space tensor
    t : the time added into the model
    x_shape: the input shape
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # based on the time pointed by, to get the result, out.shape = tensor(80,)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    # [1] * (len(x_shape) - 1)创建了一个列表，列表的长度为x_shape的长度减一，元素为1
    # the result shape: (80, 1, 1, 1)，即依照时间T选取的


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        # In unconditional model, beta_1:1e-4, beta_T:0.02, T:1000
        super().__init__()

        self.model = model
        self.T = T

        # 利用线性取点，结合论文参数，设置β参数
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # α=1-β
        alphas = 1. - self.betas
        # 计算累乘的α
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 根据t时刻的x计算公式，计算相关参数，分别是根号下的累乘以及根号下的1-累乘
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        # 生成80个随机均匀分布的正整数作为本batch的（即共选择1200w个时间）
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        size = x_0.shape[0]
        # 生成高斯正态分布的噪音
        noise = torch.randn_like(x_0)
        # 利用搜集方法搜集随机生成的t个时刻相关参数，根据公式计算t时刻的x值
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # 通过输入最后时刻的噪声与时间t，得到网络反向传播的噪声，与添加的已知噪声一道计算损失
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        # beta_1：1e-4; beta_T:0.02; T:1000
        super().__init__()

        self.model = model
        self.T = T

        # 利用随机采样获取特定时间的β信息
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        # 获取α信息
        alphas = 1. - self.betas
        # 获取累乘的α信息
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 用1填充累乘的α最后一个维度，然后再取填充后变量的前T个元素
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # 计算最外层的分母
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        # 不包括t时刻的与ε的系数喵
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # β为公式中的1-α，这里为计算方差的公式喵
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # 根据当前时间步的噪声估计均值（即μ参数）
        assert x_t.shape == eps.shape
        a = (
                extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # 计算给定时间步的均值与方差
        # below: only log_variance is used in the KL computations
        # 保证时间为0时不存在后验方差
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        # 挖掘对应时间维度的方差
        var = extract(var, t, x_t.shape)


        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        Args：
        x_T: tensor of noisy image
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            # 逆序时间步数，将扩散过程进行还原
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            # 添加随机噪声保证生成有效的图片
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # 限制x_0的范围
        return torch.clip(x_0, -1, 1)   


