import torch
from PIL import Image
from utils import set_random_seed
from collections import namedtuple

from utils import to_tensor, to_pil
from .optim_utils import (
    get_watermarking_pattern,
    get_watermarking_mask,
    inject_watermark,
    eval_watermark,
)
from guided_diffusion import guided_ddim_sample


# Generate a message (which is the key in tree-ring's paper) with a specific message seed
def generate_guided_tree_ring_message(
    message_seed, image_size, tree_ring_paras, device
):
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", list(tree_ring_paras.keys()) + ["w_seed"])(
        **tree_ring_paras, w_seed=message_seed
    )
    shape = (1, 3, image_size, image_size)
    # Generate the message, which is the key in tree-ring's paper
    message = get_watermarking_pattern(None, tree_ring_args, device, shape)
    # Message's shape is (1, 3, image_size, image_size)
    return message


# Generate a key (which is the mask in tree-ring's paper) with a specific key seed
def generate_guided_tree_ring_key(key_seed, image_size, tree_ring_paras, device):
    # For tree-ring, the key (watermarking mask) is not randomized and fully determined by the w_radius and image size
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
    shape = (1, 3, image_size, image_size)
    # In get_watermarking_mask, only the shape of init_latents_w matters, not its values
    # So we can just use random values
    init_latents_w = torch.randn(*shape, device=device)
    # Generate the key, which is the mask in tree-ring's paper
    key = get_watermarking_mask(init_latents_w, tree_ring_args, device=device)
    # Key's shape is (1, 3, image_size, image_size)
    return key


# Guided diffusion with watermark
def guided_ddim_sample_with_tree_ring(
    model,
    diffusion,
    labels,
    keys,
    messages,
    tree_ring_paras,
    image_size,
    diffusion_seed,
    progressive=False,
    return_image=False,
):
    # Diffusion seed is the random seed for diffusion sampling
    set_random_seed(diffusion_seed)
    # Assert key and message are on the same device as the model
    assert keys.device == messages.device == next(model.parameters()).device
    # Can either use the same key or message for all images, or use different keys or messages for different images
    if keys.size()[0] == 1:
        keys = keys.repeat(len(labels), 1, 1, 1)
    if messages.size()[0] == 1:
        messages = messages.repeat(len(labels), 1, 1, 1)
    assert keys.size() == messages.size() == (len(labels), 3, image_size, image_size)

    # Device and shape
    device = next(model.parameters()).device
    shape = (len(labels), 3, image_size, image_size)
    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)
    # The random initial latent is determined by the diffusion seed, so no need to keep it
    init_latents_wo = torch.randn(*shape, device=device)
    # Inject watermark
    init_latents_w = inject_watermark(init_latents_wo, keys, messages, tree_ring_args)
    # Guided diffusion with injected latent
    return guided_ddim_sample(
        model,
        diffusion,
        labels,
        image_size,
        diffusion_seed,
        init_latent=init_latents_w,
        progressive=progressive,
        return_image=return_image,
    )


# Detect tree-ring watermark
def detect_guided_tree_ring(
    reversed_latents_wo,
    reversed_latents_w,
    keys,
    messages,
    tree_ring_paras,
    image_size,
):
    # Assert key and message are on the same device
    assert keys.device == messages.device
    # Check whether the inputs are PIL images
    if isinstance(reversed_latents_wo[0], Image.Image):
        reversed_latents_wo = to_tensor(reversed_latents_wo, norm_type="naive").to(
            keys.device
        )
        reversed_latents_w = to_tensor(reversed_latents_w, norm_type="naive").to(
            keys.device
        )
    else:
        reversed_latents_wo = reversed_latents_wo.to(keys.device)
        reversed_latents_w = reversed_latents_w.to(keys.device)

    # Can either use the same key or message for all images, or use different keys or messages for different images
    max_length = max(reversed_latents_wo.size()[0], reversed_latents_w.size()[0])
    if keys.size()[0] == 1:
        keys = keys.repeat(max_length, 1, 1, 1)
    if messages.size()[0] == 1:
        messages = messages.repeat(max_length, 1, 1, 1)
    assert keys.size() == messages.size() == (max_length, 3, image_size, image_size)

    # Pack dict into namedtuple so that it is compatible with args
    tree_ring_args = namedtuple("Args", tree_ring_paras.keys())(**tree_ring_paras)

    # Evaluation by measuring the L1 distance to the true message under key
    distances_wo, distances_w = eval_watermark(
        reversed_latents_wo, reversed_latents_w, keys, messages, tree_ring_args
    )

    return distances_wo, distances_w
