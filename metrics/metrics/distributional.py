import os
import tempfile
import torch
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from PIL import Image
from .clean_fid import fid


def save_single_image_to_temp(i, image, temp_dir):
    save_path = os.path.join(temp_dir, f"{i}.png")
    image.save(save_path, "PNG")


def save_images_to_temp(images, num_workers, verbose=False):
    assert isinstance(images, list) and isinstance(images[0], Image.Image)
    temp_dir = tempfile.mkdtemp()

    # Using ProcessPoolExecutor to save images in parallel
    func = partial(save_single_image_to_temp, temp_dir=temp_dir)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = executor.map(func, range(len(images)), images)
        list(tasks) if not verbose else list(
            tqdm(
                tasks,
                total=len(images),
                desc="Saving images ",
            )
        )
    return temp_dir


# Compute FID between two sets of images
def compute_fid(
    images1,
    images2,
    mode="legacy",
    device=None,
    batch_size=64,
    num_workers=None,
    verbose=False,
):
    # Support four types of FID scores
    assert mode in ["legacy", "clean", "clip"]
    if mode == "legacy":
        mode = "legacy_pytorch"
        model_name = "inception_v3"
    elif mode == "clean":
        mode = "clean"
        model_name = "inception_v3"
    elif mode == "clip":
        mode = "clean"
        model_name = "clip_vit_b_32"
    else:
        assert False

    # Set up device and num_workers
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    if num_workers is not None:
        assert 1 <= num_workers <= os.cpu_count()
    else:
        num_workers = max(torch.cuda.device_count() * 4, 8)

    # Check images, can be paths or lists of PIL images
    if not isinstance(images1, list):
        assert isinstance(images1, str) and os.path.exists(images1)
        assert isinstance(images2, str) and os.path.exists(images2)
        path1 = images1
        path2 = images2
    else:
        assert isinstance(images1, list) and isinstance(images1[0], Image.Image)
        assert isinstance(images2, list) and isinstance(images2[0], Image.Image)
        # Save images to temp dir if needed
        path1 = save_images_to_temp(images1, num_workers=num_workers, verbose=verbose)
        path2 = save_images_to_temp(images2, num_workers=num_workers, verbose=verbose)
        
    # Attempt to cache statistics for path1
    if not fid.test_stats_exists(name=str(os.path.abspath(path1)).replace("/", "_"), mode=mode, model_name=model_name):
        fid.make_custom_stats(
            name=str(os.path.abspath(path1)).replace("/", "_"),
            fdir=path1,
            mode=mode,
            model_name=model_name,
            device=device,
            num_workers=num_workers,
            verbose=verbose,
        )
    fid_score = fid.compute_fid(
        path2,
        dataset_name=str(os.path.abspath(path1)).replace("/", "_"),
        dataset_split="custom",
        mode=mode,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )
    return fid_score
