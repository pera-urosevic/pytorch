import os
import random
from matplotlib import pyplot as plt
import torch
from PIL import Image
from config import device
from data import get_transform


def get_all_files(root_dir):
    all_files = []
    for root_dir, _, files in os.walk(root_dir):
        for filename in files:
            full_path = os.path.join(root_dir, filename)
            all_files.append(full_path)
    return all_files


def predicts(model_name, collection_name):
    model = torch.load(f"./~model/{model_name}.pt")
    model.eval()
    model.to(device)

    classes_map = torch.load(f"./~data/{collection_name}/classes_map.pt")
    transform = get_transform()
    all_files = get_all_files(f"./~data/{collection_name}")

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    with torch.no_grad():
        for i in range(cols):
            for j in range(rows):
                ax = axes[j, i]
                filename = random.choice(all_files)
                image = Image.open(filename)
                img = transform(image).unsqueeze(0)
                img = img.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                label = classes_map[predicted.item()]
                dir = os.path.basename(os.path.dirname(filename))
                file = os.path.basename(filename)
                ax.set_title(f"File: {dir}/{file}\nPredicted: {label}", loc="left")
                ax.set_axis_off()
                ax.imshow(image)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"~train/{model_name}-{collection_name}.png")
        plt.show()
