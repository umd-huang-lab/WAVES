import os
import click
import PIL
import warnings
import tempfile
from tqdm.auto import tqdm
import dotenv
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    chmod_group_write,
    parse_image_dir_path,
    save_json,
)
from utils import to_tensor

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")
if "HF_HOME" not in os.environ:
    temp_dir = tempfile.mkdtemp(prefix="huggingface_cache")
    os.environ["HF_HOME"] = temp_dir


def get_indices(path, quiet, subset, limit, subset_limit):
    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    reversed_latents_existences = check_file_existence(
        path, name_pattern="{}_reversed.pkl", limit=limit
    )
    if not quiet:
        print(
            f"Found {sum(image_existences)} images, and {sum(reversed_latents_existences)} reversed latents"
        )
        print("HF_HOME cache directory is set to:", os.environ["HF_HOME"])
    indices = existence_to_indices(
        existence_operation(
            image_existences, reversed_latents_existences, op="difference"
        ),
        limit=limit if not subset else subset_limit,
    )
    return indices


def init_model(device):
    import torch
    import diffusers
    from diffusers import DPMSolverMultistepScheduler
    from tree_ring import InversableStableDiffusionPipeline

    diffusers.utils.logging.set_verbosity_error()
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        ),
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def reverse_image(pipe, path, idx, device, torch_save):
    num_inference_steps = 50
    tester_prompt = ""  # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    image = PIL.Image.open(os.path.join(path, f"{idx}.png"))
    image_transformed = to_tensor([image]).to(text_embeddings.dtype).to(device)
    image_latents = pipe.get_image_latents(image_transformed, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        verbose=False,
    )
    torch_save(reversed_latents, os.path.join(path, f"{idx}_reversed.pkl"))
    chmod_group_write(os.path.join(path, f"{idx}_reversed.pkl"))


def worker(gpu, path, indices, counter, lock, torch_save):
    device = f"cuda:{gpu}"
    pipe = init_model(device)
    for idx in indices:
        reverse_image(pipe, path, idx, device, torch_save)
        with lock:
            counter.value += 1


def process(indices, path, quiet):
    from torch.cuda import device_count
    from torch import save as torch_save
    import torch.multiprocessing as mp
    from multiprocessing import Manager

    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    num_gpus = device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    chunk_size = len(indices) // num_gpus

    with Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        processes = []

        for gpu in range(num_gpus):
            start_idx = gpu * chunk_size
            end_idx = None if gpu == num_gpus - 1 else (gpu + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    gpu,
                    path,
                    indices[start_idx:end_idx],
                    counter,
                    lock,
                    torch_save,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(indices), desc="Reverse diffusion images", unit="img"
        ) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break

        for p in processes:
            p.join()


def report(path, quiet, limit):
    reversed_latents_existences = check_file_existence(
        path, name_pattern="{}_reversed.pkl", limit=limit
    )
    data = {}
    for i in range(limit):
        data[str(i)] = reversed_latents_existences[i]
    json_path = os.path.join(
        os.environ.get("RESULT_DIR"),
        str(path).split("/")[-2],
        f"{str(path).split('/')[-1]}-reverse.json",
    )
    save_json(data, json_path)
    if not quiet:
        print(f"Reversed latents status saved to {json_path}")


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(path, dry, subset, quiet, limit=LIMIT, subset_limit=SUBSET_LIMIT):
    _, _, _, source_name = parse_image_dir_path(path, quiet=quiet)
    # Reverse is only required for real and tree_ring watermarked images
    if not source_name in ["real", "tree_ring", "real_tree_ring"]:
        if not quiet:
            print(
                f"Reverse diffusion is only required for real and tree_ring watermarked images, not {source_name}, exiting"
            )
        return
    if not dry:
        indices = get_indices(path, quiet, subset, limit, subset_limit)
        if len(indices) == 0:
            if not quiet:
                print("All reversed latents requested already exist")
        else:
            process(indices, path, quiet)
    report(path, quiet, limit=limit)


if __name__ == "__main__":
    main()
