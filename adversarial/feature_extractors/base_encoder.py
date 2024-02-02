import torch


class BaseEncoder(torch.nn.Module):
    def forward(self, images):
        raise NotImplementedError("This method should be implemented by subclasses.")
