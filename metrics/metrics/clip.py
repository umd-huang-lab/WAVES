import torch
from PIL import Image
import open_clip


def load_open_clip_model_preprocess_and_tokenizer(device=torch.device("cuda")):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s12b_b42k", device=device
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-g-14")
    return clip_model, clip_preprocess, clip_tokenizer


def compute_clip_score(
    images,
    prompts,
    models,
    device=torch.device("cuda"),
):
    clip_model, clip_preprocess, clip_tokenizer = models
    with torch.no_grad():
        tensors = [clip_preprocess(image) for image in images]
        image_processed_tensor = torch.stack(tensors, 0).to(device)
        image_features = clip_model.encode_image(image_processed_tensor)

        encoding = clip_tokenizer(prompts).to(device)
        text_features = clip_model.encode_text(encoding)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1).cpu().numpy().tolist()
