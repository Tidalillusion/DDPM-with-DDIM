import torch


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


def generalized_steps(x, seq, model, betas, eta):
    with torch.no_grad():
        seq_next = [-1] + list(seq[:-1])
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            """To prevent the overflow of the list"""
            if j == -1 and i == 1:
                x = xt_next
                break
            if j == -1 and i != 1:
                j = 1
            print(f"The skipped step: {i}")
            t = x.new_ones([x.shape[0], ], dtype=torch.long) * i
            t_next = x.new_ones([x.shape[0], ], dtype=torch.long) * j

            """
            This part is to calculate the parameter and result of DDIM
            according to Song Jiaming. And how to calculate you can find
            in https://arxiv.org/abs/2010.02502. If you're good at Chinese,
            this blog could be a good choice: https://zhuanlan.zhihu.com/p/666552214.
            """
            x_t_alphas_bar = extract(alphas_bar, t, x.shape)
            x_t_next_alphas_bar = extract(alphas_bar, t_next, x.shape)
            model = model.to("cuda:0")
            et = model(x, t)           # By Song, et=0 or et=variance in DDPM is good choice
            mean = (x - et * (1 - x_t_alphas_bar).sqrt()) / x_t_alphas_bar.sqrt()
            c1 = eta * ((1 - x_t_alphas_bar / x_t_next_alphas_bar)
                        * (1 - x_t_next_alphas_bar) / (1 - x_t_alphas_bar)).sqrt()
            c2 = ((1 - x_t_next_alphas_bar) - c1 ** 2).sqrt()
            xt_next = x_t_next_alphas_bar.sqrt() * mean + c1 * torch.randn_like(x) + c2 * et
            x = xt_next
        x_0 = x
    return torch.clip(x_0, -1, 1)
