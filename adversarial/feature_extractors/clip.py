import torch
from transformers import AutoProcessor, CLIPModel
from .base_encoder import BaseEncoder
from torchvision import transforms

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class ClipEmbedding(BaseEncoder):
    def __init__(self):
        super(ClipEmbedding, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        outputs = self.model.get_image_features(**inputs)
        pooled_output = outputs
        return pooled_output
