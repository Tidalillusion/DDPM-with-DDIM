import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    ARGS：
    V ： the tensor that sqrt or 1 - sqrt； and the sqrt is the linespace tensor
    t : the time added into the model
    x_shape: the input shape
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # based on the time pointed by, to get the result, eg: out.shape = tensor(80,)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    # the result shape: (80, 1, 1, 1)，which is chosen by random time from radiant function


class GaussianDiffusionTrainer(nn.Module):
    """This part, U are suggested to get how to calulate in https://arxiv.org/abs/2006.11239"""
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # set the beta by the way of linear
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # According to Ho's paper, α=1-β
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 根据t时刻的x计算公式，计算相关参数，分别是根号下的累乘以及根号下的1-累乘
        # In the t step. to calculate the parameter
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        # The number of t equals to the batch size, and I guess this is because of speed
        # and based on the central limit theorem
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        size = x_0.shape[0]
        # The initial noise U input
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # The loss is between the noise U input and the noise model predicting
        # According to Ho, the prediction of noise is best in that time and this experiment
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        # get the β based on the given time step
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        # get the α
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # by adding the value 1 to the front of the list, get the previous α_bar values to go on
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        """these can be found in the Ho's paper"""
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # calculate the mean value of the distribution
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
        # get the mean and variance of the distribution
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        # get the variance according to the given time step
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
            # reversed the time step to get the distribution from the input Gaussian noise
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            # add the Gaussian noise to get more diverse distribution
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   
