import os
import torch
from .script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


# Diffusion model parameters
guided_diffusion_64x64_paras = dict(
    image_size=64,
    num_channels=192,
    num_res_blocks=3,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    dropout=0.1,
    class_cond=True,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=True,
    learn_sigma=True,
    diffusion_steps=1000,
    noise_schedule="cosine",
    timestep_respacing="ddim50",
    use_scale_shift_norm=True,
)
guided_diffusion_256x256_paras = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    class_cond=True,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
    learn_sigma=True,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="ddim50",
    use_scale_shift_norm=True,
)


# Get the default parameters for guided diffusion
def get_default_guided_diffusion_paras(image_size):
    # Support two image sizes
    assert image_size in [64, 256]
    if image_size == 64:
        return guided_diffusion_64x64_paras
    else:
        return guided_diffusion_256x256_paras


# Load guided diffusion model and weights
def load_guided_diffusion_model(image_size, device):
    # Support two image sizes
    assert image_size in [64, 256]
    paras = model_and_diffusion_defaults()
    # Update with default parameters, see https://github.com/openai/guided-diffusion
    paras.update(get_default_guided_diffusion_paras(image_size))
    # Initilaize model and load weights
    model, diffusion = create_model_and_diffusion(**paras)
    model.load_state_dict(
        torch.load(
            os.path.join(
                os.environ.get("MODEL_DIR"),
                f"guided-diffusion/{image_size}x{image_size}_diffusion.pt",
            ),
            map_location=device,
        )
    )
    model.to(device)
    # Convert to FP16
    if paras["use_fp16"]:
        model.convert_to_fp16()
    # Set eval flag
    model.eval()
    return model, diffusion
