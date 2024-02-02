import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import json

from utils import to_tensor
from .exp_utils import set_random_seed


# Get ImageNet class names
def get_imagenet_class_names(labels=None):
    with open(
        os.path.join(
            os.environ.get("DATASET_DIR"), "source/imagenet/imagenet_class_index.json"
        ),
        "r",
    ) as file:
        cid_to_wnid_and_words = json.load(file)
    if labels is None:
        return {int(cid): words for cid, (wnid, words) in cid_to_wnid_and_words.items()}
    else:
        return [cid_to_wnid_and_words[str(cid)][1] for cid in labels]


# Get ImageNet wnids
def get_imagenet_wnids(labels=None):
    with open(
        os.path.join(
            os.environ.get("DATASET_DIR"), "source/imagenet/imagenet_class_index.json"
        ),
        "r",
    ) as file:
        cid_to_wnid_and_words = json.load(file)
    if labels is None:
        return {int(cid): wnid for cid, (wnid, words) in cid_to_wnid_and_words.items()}
    else:
        return [cid_to_wnid_and_words[str(cid)][0] for cid in labels]


# Load imagenet subset and class names
def load_imagenet_subset(dataset_name, convert_to_tensor=True, norm_type="naive"):
    assert dataset_name in ["Tiny-ImageNet", "Imagenette"]
    # Load WordNet IDs and class names
    with open(
        os.path.join(
            os.environ.get("DATASET_DIR"), "source/imagenet/tiny-imagenet-200/wnids.txt"
        ),
        "r",
    ) as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_words = {}
    with open(
        os.path.join(
            os.environ.get("DATASET_DIR"), "source/imagenet/tiny-imagenet-200/words.txt"
        ),
        "r",
    ) as f:
        for line in f.readlines():
            wnid, words = line.strip().split("\t")
            wnid_to_words[wnid] = words
    # Tiny-ImageNet
    if dataset_name == "Tiny-ImageNet":
        # Tiny-ImageNet dataset
        data_dir = os.path.join(
            os.environ.get("DATASET_DIR"),
            "source/imagenet/tiny-imagenet-200",
        )
        dataset = datasets.ImageFolder(
            f"{data_dir}/train",
            lambda x: to_tensor([x], norm_type=norm_type) if convert_to_tensor else x,
        )
        assert len(dataset) == 100000
        # Tiny-ImageNet class names
        class_names = [wnid_to_words[wnid] for wnid in sorted(wnids)]
        assert len(class_names) == 200
    # Imagenette
    elif dataset_name == "Imagenette":
        # Imagenette dataset
        data_dir = os.path.join(
            os.environ.get("DATASET_DIR"),
            "source/imagenet/imagenette2-320",
        )
        dataset = datasets.ImageFolder(
            f"{data_dir}/train",
            transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(256),
                    lambda x: to_tensor([x], norm_type=norm_type)
                    if convert_to_tensor
                    else x,
                ]
            ),
        )
        assert len(dataset) == 9469
        # Imagenette class names
        class_names = [wnid_to_words[wnid] for wnid in dataset.classes]
        assert len(class_names) == 10
    return dataset, class_names


# Sample train and test sets from a dataset
def sample_train_and_test_sets(train_size, test_size, dataset, exp_rand_seed=None):
    assert (train_size + test_size) <= len(dataset)
    if exp_rand_seed is not None:
        set_random_seed(exp_rand_seed)
        # Random sample without replacement
        indices = torch.randperm(len(dataset))[: train_size + test_size]
    else:
        # Sequential sample
        indices = torch.arange(train_size + test_size, dtype=torch.long)
    images = [dataset[idx][0] for idx in indices]
    labels = [dataset[idx][1] for idx in indices]
    return (
        images[:train_size],
        labels[:train_size],
        images[-test_size:],
        labels[-test_size:],
    )


# Load imagenet guided diffusion generated images and class names
def load_imagenet_guided(
    image_size, dataset_template, convert_to_tensor=True, norm_type="naive"
):
    assert image_size in [64, 256]
    assert dataset_template in ["Tiny-ImageNet", "Imagenette"]
    # Load WordNet IDs and class names
    wnid_to_words = {}
    with open(
        os.path.join(
            os.environ.get("DATASET_DIR"), "source/imagenet/tiny-imagenet-200/words.txt"
        ),
        "r",
    ) as f:
        for line in f.readlines():
            wnid, words = line.strip().split("\t")
            wnid_to_words[wnid] = words
    # Load dataset
    data_dir = os.path.join(
        os.environ.get("DATASET_DIR"),
        f"source/generated/imagenet_guided_{image_size}_{dataset_template.lower()}",
    )

    dataset = datasets.ImageFolder(
        f"{data_dir}/train",
        lambda x: to_tensor([x], norm_type=norm_type) if convert_to_tensor else x,
    )
    # ImageNet class names
    class_names = [wnid_to_words[wnid] for wnid in dataset.classes]
    # Check sizes
    assert len(set(label for _, label in dataset)) == len(class_names)

    return dataset, class_names


# Sample images by condition on labels from a dataset
def sample_images_by_label_cond(
    dataset, num_samples, label_cond, replace=False, sampling_seed=None
):
    if sampling_seed is not None:
        set_random_seed(sampling_seed)
    # Determine the samples where the condition is met
    if callable(label_cond):  # If y_condition is a lambda or function
        indices = [i for i, (_, label) in enumerate(dataset) if label_cond(label)]
    else:  # If y_condition is a numeric value
        indices = [i for i, (_, label) in enumerate(dataset) if label == label_cond]

    # If no samples meet the condition, raise an error
    if not indices:
        assert False
    # If more samples are requested than available and replace=False, raise an error
    if not replace and num_samples > len(indices):
        assert False

    # Sample indices
    sampled_indices = np.random.choice(
        indices, size=num_samples, replace=replace
    ).tolist()

    # Fetch images and labels
    images = [dataset[idx][0] for idx in sampled_indices]
    labels = [dataset[idx][1] for idx in sampled_indices]
    return images, labels


# Sample images by a set of labels from a dataset
def sample_images_by_label_set(
    dataset, num_sample_per_class, label_cond=None, replace=False, sampling_seed=None
):
    if sampling_seed is not None:
        set_random_seed(sampling_seed)
    # If label_cond is None, collect all unique labels from the dataset
    if label_cond is None:
        label_cond = set([label for _, label in dataset])
    # If label_cond is a list or set, convert to a set for O(1) lookups
    elif not callable(label_cond):
        label_cond = set(label_cond)

    # Map from label to list of dataset indices with that label
    label_to_indices = {}
    for i, (_, label) in enumerate(dataset):
        if label in label_cond if not callable(label_cond) else label_cond(label):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)

    # Calculate samples based on num_sample_per_class
    sampled_indices = []
    for label_indices in label_to_indices.values():
        if len(label_indices) < num_sample_per_class and not replace:
            assert False
        sampled_indices.extend(
            np.random.choice(
                label_indices, num_sample_per_class, replace=replace
            ).tolist()
        )

    # Fetch images and labels
    images = [dataset[idx][0] for idx in sampled_indices]
    labels = [dataset[idx][1] for idx in sampled_indices]
    return images, labels
