import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


def data_load_x():
    return torch.load("./~data/x.pt")


def data_load_y():
    return torch.load("./~data/y.pt")


def data_load_x_train():
    return torch.load("./~data/x_train.pt")


def data_load_x_test():
    return torch.load("./~data/x_test.pt")


def data_load_y_train():
    return torch.load("./~data/y_train.pt")


def data_load_y_test():
    return torch.load("./~data/y_test.pt")
