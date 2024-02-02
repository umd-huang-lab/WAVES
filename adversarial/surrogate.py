import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import os
import argparse
from torchvision.utils import save_image
from torchvision.models import resnet18
from torchattacks.attack import Attack
from torch.utils.data import Dataset, DataLoader


EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 4


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PGD Attack on the surrogate clasifier."
    )
    parser.add_argument(
        "--attack_name",
        type=str,
        default="unwm_wm",
        choices=["unwm_wm", "real_wm", "wm1_wm2"],
        help="Three adversarial surrogate detector attacks tested in the paper.",
    )
    parser.add_argument(
        "--watermark_name",
        type=str,
        default="tree_ring",
        choices=["tree_ring", "stable_sig", "stegastamp"],
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2,
        choices=[2, 4, 6, 8],
        help="The perturbation radius of adversarial attacks. It will be divided by 255 in the code.",
    )
    parser.add_argument(
        "--target_label",
        type=int,
        default=0,
        choices=[0, 1],
        help="The target label for PGD targeted-attack. Labels are the ones used in surrogate model training. "
        "For umwm_wm, 0 is non-watermarked, 1 is watermarked. To remove watermarks, the target_label should be 0.",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def adv_surrogate_model_attack(
    data_path,
    model_path,
    strength,
    output_path,
    target_label,
    warmup=True,
    device=torch.device("cuda:0"),
):
    # check if the file/directory paths exist
    for path in [data_path, model_path, output_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")
    # check if the target_label is in {0, 1}
    if target_label not in {0, 1}:
        raise ValueError("target_label must be 0 or 1.")

    # load surrogate model
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification: 2 classes
    save_path_full = os.path.join(model_path)
    model.load_state_dict(torch.load(save_path_full))
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # load data
    transform = transforms.ToTensor()
    dataset = SimpleImageFolder(data_path, transform=transform)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    print("Data loaded!")

    # warm up
    if warmup:
        # Warmup to get the average delta for the attack
        average_delta_list = []
        attack_warmup = pgd_attack_classifier(
            model=model,
            eps=EPS_FACTOR * strength,
            alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
            steps=N_STEPS,
            random_start=True,
        )
        for i, (images, image_paths) in enumerate(train_loader):
            images = images.to(device)
            if target_label == 1:
                target_labels = torch.ones(images.size(0), dtype=torch.long).to(device)
            elif target_label == 0:
                target_labels = torch.zeros(images.size(0), dtype=torch.long).to(device)

            # Attack images
            images_adv = attack_warmup(images, target_labels, init_delta=None)

            average_delta_list.append((images_adv - images).mean(dim=0))

            if i >= 20:
                break

        average_delta = torch.cat(average_delta_list, dim=0).mean(dim=0)

        print("Warmup finished!")
    else:
        average_delta = None

    # Generate adversarial images
    attack = pgd_attack_classifier(
        model=model,
        eps=EPS_FACTOR * strength,
        alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
        steps=N_STEPS,
        random_start=False if warmup else True,
    )
    for i, (images, image_paths) in enumerate(test_loader):
        images = images.to(device)
        if target_label == 1:
            target_labels = torch.ones(images.size(0), dtype=torch.long).to(device)
        elif target_label == 0:
            target_labels = torch.zeros(images.size(0), dtype=torch.long).to(device)

        # PGD attack
        images_adv = attack(images, target_labels, init_delta=average_delta)

        # save images
        for img_adv, image_path in zip(images_adv, image_paths):
            save_path = os.path.join(output_path, os.path.basename(image_path))
            save_image(img_adv, save_path)

    print("Attack finished!")
    return


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=None):
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.root = root
        self.transform = transform
        self.extensions = extensions

        # Load filenames from the root
        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in self.extensions
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path  # return image path to identify the image file later

    def __len__(self):
        return len(self.filenames)


class WarmupPGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels, init_delta=None):
        """
        Overridden.
        """
        self.model.eval()

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def pgd_attack_classifier(model, eps, alpha, steps, random_start=True):
    # Create an instance of the attack
    attack = WarmupPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
    )

    # Set targeted mode
    attack.set_mode_targeted_by_label(quiet=True)

    return attack
