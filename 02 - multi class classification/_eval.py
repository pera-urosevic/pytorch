from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from _data import (
    data_load_species,
    data_load_x,
    data_load_y,
)
from _model import model_predict, model_accuracy_score


def eval_model(model, x, y):
    predicted, labels = model_predict(model, x, y)
    return predicted, labels


def eval_model_accuracy(predicted, labels):
    accuracy = model_accuracy_score(predicted, labels)
    print(f"Accuracy: {accuracy:.4f}")


def eval_model_plot(species, predicted, labels):
    cm = confusion_matrix(labels, predicted)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.xticks(np.arange(len(np.unique(labels))), species)
    plt.yticks(np.arange(len(np.unique(labels))), species)
    plt.tight_layout()
    plt.show()


def eval_model_unseen(model, x, y):
    species = data_load_species()
    with torch.no_grad():
        p = model(x)
        print(f"Unseen prediction tensor: {p}")
        predicted = species[p.argmax()]
        print(f"Unseen predicted species: {predicted}")
        print(f"Unseen actual species: {y}")
