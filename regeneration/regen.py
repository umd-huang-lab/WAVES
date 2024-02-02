from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import ColorJitter
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
)
import glob
import numpy as np
import argparse
from diffusers import ReSDPipeline


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, strength=1, metric="mse", device="cpu"):
        if model_name == "bmshj2018-factorized":
            self.model = (
                bmshj2018_factorized(quality=strength, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "bmshj2018-hyperprior":
            self.model = (
                bmshj2018_hyperprior(quality=strength, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "mbt2018-mean":
            self.model = (
                mbt2018_mean(quality=strength, pretrained=True).eval().to(device)
            )
        elif model_name == "mbt2018":
            self.model = mbt2018(quality=strength, pretrained=True).eval().to(device)
        elif model_name == "cheng2020-anchor":
            self.model = (
                cheng2020_anchor(quality=strength, pretrained=True).eval().to(device)
            )
        else:
            raise ValueError("model name not supported")
        self.device = device

    def attack(self, image, device):

        img = image.convert("RGB")
        img = img.resize((512, 512))
        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        out = self.model(img)
        out["x_hat"].clamp_(0, 1)
        out = transforms.ToPILImage()(out["x_hat"].squeeze().cpu())
        return out


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, noise_step=60, captions={}):
        self.pipe = pipe
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(
            f"Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}"
        )

    def attack(self, image, device, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor(
                [self.noise_step], dtype=torch.long, device=self.device
            )
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(
                    prompts_buf,
                    head_start_latents=latents,
                    head_start_step=50 - max(self.noise_step // 20, 1),
                    guidance_scale=7.5,
                    generator=generator,
                )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    return img

            img = np.asarray(image) / 255
            img = (img - 0.5) * 2
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(
                img.to(device=torch.device("cuda"), dtype=torch.float16)
            ).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn(
                [1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device
            )

            latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(
                torch.half
            )
            latents_buf.append(latents)
            outs_buf.append("")
            prompts_buf.append("")

            img = batched_attack(latents_buf, prompts_buf, outs_buf)
            return img


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Define a custom argument type for a list of strings
    def list_of_attacks(arg):
        return arg.split(",")

    # List of attacks
    parser.add_argument("--attack_list", type=list_of_attacks)

    parser.add_argument(
        "--attack_method",
        type=str,
        choices=["regen_vae", "regen_diffusion"],
        default="regen_vae",
        help="Attacking method.",
    )
    # attacking methods' specific hyperparameters
    parser.add_argument(
        "--vae_name",
        type=str,
        default="bmshj2018-factorized",
        choices=[
            "bmshj2018-factorized",
            "cheng2020-anchor",
            "bmshj2018-hyperprior",
            "mbt2018-mean",
            "mbt2018",
        ],
        help="The VAE model name. All the models are loaded from compressai.",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def remove_watermark(attack_method, image, strength, model, device):
    # create attacker
    print(f"Creating attacker {attack_method}...")
    if attack_method == "regen_vae":
        attacker = VAEWMAttacker(model, quality=strength, metric="mse", device=device)

    elif attack_method == "regen_vae":
        attacker = VAEWMAttacker(model, quality=strength, metric="mse", device=device)

    elif attack_method == "regen_diffusion":
        pipe = ReSDPipeline.from_pretrained(
            model, torch_dtype=torch.float16, revision="fp16"
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        attacker = DiffWMAttacker(pipe, noise_step=strength, captions={})

    else:
        raise Exception(f"Unknown attacking method: {attack_method}!")

    img = attacker.attack(image, device)

    return img


def regen_diff(
    image, strength, model="CompVis/stable-diffusion-v1-4", device=torch.device("cpu")
):
    image = remove_watermark("regen_diffusion", image, strength, model, device)
    return image


def rinse_2xDiff(image, strength, model="", device=torch.device("cpu")):
    first_attack = True
    for attack in ["regen_diffusion", "regen_diffusion"]:
        if first_attack:
            image = remove_watermark(attack, image, strength, model, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, model, device)
    return image


def rinse_4xDiff(image, strength, model="", device=torch.device("cpu")):
    first_attack = True
    for attack in [
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
    ]:
        if first_attack:
            image = remove_watermark(attack, image, strength, model, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, model, device)
    return image


def regen_vae(
    image, strength, model="bmshj2018-factorized", device=torch.device("cpu")
):
    image = remove_watermark("regen_vae", image, strength, model, device)
    return image
