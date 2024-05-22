from Diffusion.Train import train, eval
from Diffusion.DDIM import eval_ddim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(model_config = None):
    modelConfig = {
        "state": "eval_ddim",    # or eval or eval_ddim or train
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "sample_type": "generalized",    # or ddpm_noisy
        "skip_type": "uniform",    # or quard
        "sikpped_step": 250,     # or 100, 250, 200
        "noise_type": "sample_sequence",     # or sample_fid, sample_interpolation
        "ddim_sampled_dir": "./ddim_SampledImgs/",
        "ddim_sampledImgName": "ddim_SampledNoGuidenceImgs.png",
        "ddim_sampledNoisyImgName": "ddim_SampledNoGuidenceImgs.png",
        "eta": 0
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    if modelConfig["state"] == "eval_ddim":
        eval_ddim(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
