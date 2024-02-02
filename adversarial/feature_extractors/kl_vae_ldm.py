import torch
from .base_encoder import BaseEncoder
from omegaconf import OmegaConf
import importlib


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class KLVAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = self.get_model(model_name)

    def load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(
            ckpt,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)

        delattr(model, "decoder")

        return model

    def get_model(self, name):
        config_path = "./models/ldm/" + name + "/config.yaml"
        model_path = "./models/ldm/" + name + "/model.ckpt"
        config = OmegaConf.load(config_path)
        model = self.load_model_from_config(config, model_path)
        return model

    def forward(self, images):
        images = 2.0 * images - 1.0
        output = self.model.encode(images)
        z = output.mode()
        return z
