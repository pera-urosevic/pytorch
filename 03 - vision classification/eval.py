import matplotlib.pyplot as plt
import torch
import seaborn as sns


def eval_train(models, collections):
    for model in models:
        for collection in collections:
            data = torch.load(f"~train/{model}-{collection}.pt")
            plt.subplots(1, 2, figsize=(12, 4))
            plt.subplot(1, 2, 1)
            x = list(range(0, len(data["loss"])))
            y = data["loss"]
            plt.plot(x, y, label=f"Loss [{model}]", color="red")
            plt.subplot(1, 2, 2)
            x = list(range(0, len(data["accuracy"])))
            y = data["accuracy"]
            plt.plot(x, y, label=f"Accuracy [{model}]", color="green")
            plt.suptitle(f"Training Loss and Accuracy: {model}/{collection}")
            plt.tight_layout()
            plt.savefig(f"~train/{model}-{collection}.png")
