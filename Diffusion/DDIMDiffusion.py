import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    # based on the time pointed by, to get the result, out.shape = tensor(80,)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    # [1] * (len(x_shape) - 1)创建了一个列表，列表的长度为x_shape的长度减一，元素为1
    # the result shape: (80, 1, 1, 1)，即依照时间T选取的


def generalized_steps(x, seq, model, betas, eta):
    with torch.no_grad():
        seq_next = [-1] + list(seq[:-1])
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            if j == -1 and i == 1:
                x = xt_next
                break
            if j == -1 and i != 1:
                j = 1
            print(f"The skipped step: {i}")
            t = x.new_ones([x.shape[0], ], dtype=torch.long) * i
            t_next = x.new_ones([x.shape[0], ], dtype=torch.long) * j
            x_t_alphas_bar = extract(alphas_bar, t, x.shape)
            x_t_next_alphas_bar = extract(alphas_bar, t_next, x.shape)
            model = model.to("cuda:0")
            et = model(x, t)
            mean = (x - et * (1 - x_t_alphas_bar).sqrt()) / x_t_alphas_bar.sqrt()
            c1 = eta * ((1 - x_t_alphas_bar / x_t_next_alphas_bar)
                        * (1 - x_t_next_alphas_bar) / (1 - x_t_alphas_bar)).sqrt()
            c2 = ((1 - x_t_next_alphas_bar) - c1 ** 2).sqrt()
            xt_next = x_t_next_alphas_bar.sqrt() * mean + c1 * torch.randn_like(x) + c2 * et
            x = xt_next
        x_0 = x
    return torch.clip(x_0, -1, 1)







