from .base_encoder import BaseEncoder
import torchvision.models as models
import torch
import torchvision.transforms.functional as TF


class ResNet18Embedding(BaseEncoder):
    def __init__(self, layer):
        super().__init__()
        original_model = models.resnet18(pretrained=True)
        # Define normalization layers
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        # Extract the desired layers from the original model
        if layer == "layer1":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-6])
        elif layer == "layer2":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-5])
        elif layer == "layer3":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-4])
        elif layer == "layer4":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-3])
        elif layer == "last":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        else:
            raise ValueError("Invalid layer name")

    def forward(self, images):
        # Normalize the input
        images = TF.resize(images, [224, 224])
        images = (images - self.mean) / self.std
        return self.features(images)
