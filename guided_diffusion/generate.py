from PIL import Image
import torch
from utils import set_random_seed, to_pil, to_tensor
from .script_util import NUM_CLASSES


# Guided diffusion with DDIM sampler
def guided_ddim_sample(
    model,
    diffusion,
    labels,
    image_size,
    diffusion_seed,
    init_latent=None,
    progressive=False,
    return_image=False,
):
    # Diffusion seed is the random seed for diffusion sampling
    set_random_seed(diffusion_seed)
    # For guided diffusion, prompts are class ids
    assert isinstance(labels, list) and all(
        isinstance(label, int) and 0 <= label < NUM_CLASSES for label in labels
    )
    # Device and shape
    device = next(model.parameters()).device
    shape = (len(labels), 3, image_size, image_size)
    # The random initial latent is determined by the diffusion seed, so no need to keep it
    if init_latent is None:
        init_latent = torch.randn(*shape, device=device)
    # Diffusion
    if not progressive:
        output = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latent,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
            return_image=return_image,
        )
        return output
    else:
        output = []
        for sample in diffusion.ddim_sample_loop_progressive(
            model=model,
            shape=shape,
            noise=init_latent,
            model_kwargs=dict(y=torch.tensor(labels, device=device)),
            device=device,
        ):
            if not return_image:
                output.append(sample["sample"])
            else:
                output.append(to_pil(sample["sample"]))
        return output


# Reverse guided diffusion with DDIM sampler
def guided_reverse_ddim_sample(
    model,
    diffusion,
    images,
    image_size,
    default_labels=0,
    progressive=False,
    return_image=False,
):
    # Reverse diffusion of DDIM smapling is deterministic, so this line has no effect
    set_random_seed(0)
    # Device and shape
    device = next(model.parameters()).device
    shape = (len(images), 3, image_size, image_size)
    # If default labels is a single int, repeat it for all images
    if isinstance(default_labels, int):
        default_labels = [default_labels] * len(images)
    # Check whether the inputs are PIL images
    if isinstance(images[0], Image.Image):
        images = to_tensor(images, norm_type="naive").to(device)
    # Reversed diffusion
    if not progressive:
        output = diffusion.ddim_reverse_sample_loop(
            model=model,
            shape=shape,
            image=images,
            # Reverse diffusion does not depends on the labels, thus pass in dummy labels
            model_kwargs=dict(y=torch.tensor(default_labels, device=device)),
            device=device,
        )
        if not return_image:
            return output
        else:
            return to_pil(output)
    else:
        output = []
        for sample in diffusion.ddim_reverse_sample_loop_progressive(
            model=model,
            shape=shape,
            image=images,
            # Reverse diffusion does not depends on the labels, thus pass in dummy labels
            model_kwargs=dict(y=torch.tensor(default_labels, device=device)),
            device=device,
        ):
            if not return_image:
                output.append(sample["sample"])
            else:
                output.append(to_pil(sample["sample"]))
        return output
