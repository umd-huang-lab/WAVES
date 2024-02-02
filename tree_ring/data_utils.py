import os
import torch
from torchvision.datasets import ImageFolder
from utils import to_tensor


class MultiLabelImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MultiLabelImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        # Detect the number of labels
        example_dir = os.listdir(root)[0]
        num_labels = example_dir.count("_") + 1

        label_sets = [set() for _ in range(num_labels)]

        # Collect all unique labels for each position
        for d in os.listdir(root):
            labels = d.split("_")
            for i, label in enumerate(labels):
                label_sets[i].add(label)

        # Create label mappings for each label set
        self.label_to_idx = [
            {label: idx for idx, label in enumerate(sorted(label_set))}
            for label_set in label_sets
        ]

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)

        # Extract the multi-labels from the folder name
        dirname = os.path.basename(os.path.dirname(path))
        labels = dirname.split("_")

        label_idxs = [
            label_map[label] for label, label_map in zip(labels, self.label_to_idx)
        ]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            label_idxs = self.target_transform(label_idxs)

        return sample, tuple(label_idxs)


def load_tree_ring_guided(
    image_size,
    dataset_template,
    num_key_seeds,
    num_message_seeds,
    convert_to_tensor=True,
    norm_type="naive",
):
    assert image_size in [64, 256]
    assert dataset_template in ["Tiny-ImageNet", "Imagenette"]
    # Load WordNet IDs and class names
    wnid_to_words = {}
    with open("./datasets/tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            wnid, words = line.strip().split("\t")
            wnid_to_words[wnid] = words
    # Load dataset
    data_dir = f"./datasets/tree_ring_guided_{image_size}_{dataset_template.lower()}_{num_key_seeds}k_{num_message_seeds}m"

    dataset = MultiLabelImageFolder(
        f"{data_dir}/train",
        lambda x: to_tensor([x], norm_type=norm_type) if convert_to_tensor else x,
    )
    # ImageNet class names
    class_names = [
        wnid_to_words[class_names.split("_")[0]] for class_names in dataset.classes
    ]
    # Load keys and messages
    keys = torch.load(f"{data_dir}/keys.pt")
    messages = torch.load(f"{data_dir}/messages.pt")
    # Check sizes
    assert len(set(label[0] for _, label in dataset)) == len(class_names)
    assert len(set(label[1] for _, label in dataset)) == len(keys)
    assert len(set(label[2] for _, label in dataset)) == len(messages)

    return dataset, class_names, keys, messages
