import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def split_balanced(dataset_dir, transform, test_size=0.1, random_state=42):
    data_path = os.path.join(dataset_dir)
    image_paths = [
        os.path.join(root, filename) for root, _, filenames in os.walk(data_path) for filename in filenames
        if filename.lower().endswith(".jpg")
    ]
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    labels_map = {label: i for i, label in enumerate(set(labels))}
    labels_numerical = [labels_map[label] for label in labels]

    train_indices, test_indices, train_labels, test_labels = train_test_split(np.arange(len(image_paths)),
                                                                              labels_numerical,
                                                                              test_size=test_size,
                                                                              stratify=labels_numerical,
                                                                              random_state=random_state)

    train_dataset = Subset(datasets.ImageFolder(root=dataset_dir, transform=transform), indices=train_indices)
    test_dataset = Subset(datasets.ImageFolder(root=dataset_dir, transform=transform), indices=test_indices)

    return train_dataset, test_dataset


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def dataset(name="food"):
    root = f"./~data/{name}"
    transform = get_transform()

    train_dataset, test_dataset = split_balanced(root, transform=transform)

    print(f"Training images: {len(train_dataset)}, classes: {train_dataset.dataset.classes}")
    print(f"Testing images: {len(test_dataset)}, classes: {test_dataset.dataset.classes}")

    train_classes_map = train_dataset.dataset.classes
    test_classes_map = test_dataset.dataset.classes
    if train_classes_map != test_classes_map:
        raise ValueError(f"Train and test classes are not the same: {train_classes_map} != {test_classes_map}")
    torch.save(train_classes_map, root + "/classes_map.pt")

    return train_dataset, test_dataset


if __name__ == "__main__":

    def distribution(dataset):
        class_counts = {}
        for _, c in dataset:
            if c in class_counts:
                class_counts[c] += 1
            else:
                class_counts[c] = 1

        classes_map = dataset.dataset.classes
        for c, count in class_counts.items():
            print(f"Label {classes_map[c]}: {count} images")

    train_dataset, test_dataset = dataset('food_micro')

    print("\nTraining images distribution:")
    distribution(train_dataset)

    print("\nTesting images distribution:")
    distribution(test_dataset)
