import os
from typing import Dict

import torch
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image
from Diffusion.Model import UNet


class DDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, sample_type, skip_type, eta, sikpped_step):
        super().__init__()
        self.model = model
        self.T = T
        self.sample_type = sample_type
        self.skip_type = skip_type
        self.eta = eta
        self.sikpped_step = sikpped_step

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        betas = np.linspace(beta_1, beta_T, T, dtype=np.float64)
        self.betas = torch.from_numpy(betas).float().to('cuda:0')

    def sample_image_seq(self, noisyImage):
        """
        For now, I have just finished the code of generalized steps,
        in the mode, you can set the parameter in the file of Main to
        choose what kind of noisy schedule you want to use.
        And according to Song Jiaming, it is the mode of uniform and quard
        """
        if self.sample_type == "generalized":
            if self.skip_type == "uniform":
                skip = self.T // int(self.sikpped_step)
                seq = range(1, self.T, skip)
            if self.skip_type == "quard":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.T * 0.8), int(self.sikpped_step)
                        ) ** 2
                )
                seq = [int(s) for s in list(seq)]
            from Diffusion.DDIMDiffusion import generalized_steps
            with torch.no_grad():
                x0 = generalized_steps(noisyImage, seq, self.model, self.betas, self.eta)

        if self.sample_type == "ddpm_noisy":    # todo: what does it mean?
            if self.skip_type == "uniform":
                skip = self.T // int(self.sikpped_step)
                seq = range(1, self.T, skip)
            if self.skip_type == "quard":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.T * 0.8), int(self.sikpped_step)
                        ) ** 2
                )
                seq = [int(s) for s in list(seq)]
                from Diffusion.DDIMDiffusion import ddpm_steps
                with torch.no_grad():
                    x0, _ = ddpm_steps(noisyImage, seq, self.model, self.betas)
        return x0

    def forward(self, noisyImage):
        x0 = self.sample_image_seq(noisyImage)
        return torch.clip(x0, -1, 1)


def eval_ddim(modelConfig: Dict):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]),
                          map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = DDIM(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
                       modelConfig["sample_type"], modelConfig["skip_type"], modelConfig["eta"],
                       modelConfig["sikpped_step"])
        num_batches = modelConfig["batch_size"]
        if modelConfig["noise_type"] == "sample_sequence":
            noisyImage = sample_sequence(num_batches, device)
        if modelConfig["noise_type"] == "sample_fid":
            noisyImage = sample_fid(num_batches, device)
        # if modelConfig["noise_type"] == "sample_interpolation":还没想清楚
            # noisyImage = sample_interpolation(device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["ddim_sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5
        save_image(sampledImgs,
                   os.path.join(modelConfig["ddim_sampled_dir"], modelConfig["ddim_sampledImgName"]),
                   nrow=modelConfig["nrow"])


def sample_sequence(num_batches, input_device):
    noisyImage = torch.randn(
        size=[num_batches, 3, 32, 32], device=input_device
    )
    return noisyImage


def sample_fid(num_batches, input_device):
    noisyImage = torch.randn(
        size=[num_batches*100, 3, 32, 32], device=input_device
    )
    return noisyImage


def sample_interpolation(device):
    def slerp(z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )
    z1 = torch.randn(
        1, 3, 32, 32, device=device,
    )
    z2 = torch.randn(
        1, 3, 32, 32, device=device,
    )
    alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
    z_ = []
    for i in range(alpha.size(0)):
        # according to the tensor to interpolate
        z_.append(slerp(z1, z2, alpha[i]))
    x = torch.cat(z_, dim=0)
