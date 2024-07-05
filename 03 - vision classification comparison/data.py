import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

collections = ["mnist", "simpsons-color", "simpsons-mono"]

input_shape = 3 * 28 * 28

output_shape = 10

batch_size = 256


def data_loader(collection, type, seed=42):
    torch.manual_seed(seed)
    data_dir = f"~data/{collection}/{type}"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
