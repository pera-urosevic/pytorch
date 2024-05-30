import torch
import numpy as np
import matplotlib.pyplot as plt
from _data import (
    data_load_x,
    data_load_y,
    data_load_x_train,
    data_load_y_train,
    data_load_x_test,
    data_load_y_test,
)
from _model import model_accuracy


def eval_model(model):
    x_train = data_load_x_train()
    y_train = data_load_y_train()
    x_test = data_load_x_test()
    y_test = data_load_y_test()

    model = torch.load("./~model/model.pt")
    model.eval()

    train_accuracy = model_accuracy(model, x_train, y_train)
    test_accuracy = model_accuracy(model, x_test, y_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


def eval_model_plot(model):
    x = data_load_x()
    y = data_load_y()
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = Z.reshape(xx.shape)
        Z = Z.round().numpy()

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k", marker="o")
    plt.show()
