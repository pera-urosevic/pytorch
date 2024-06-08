import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def data_generate():
    data = pd.read_csv("~data/iris.csv")

    print(data.head())
    print()

    species = data["species"].unique()
    print(species)
    species_map = {label: i for i, label in enumerate(species)}
    data["species"] = data["species"].map(species_map)

    x = data.drop("species", axis=1).values
    y = data["species"].values

    num_classes = len(species)
    y_one_hot = torch.zeros((len(y), num_classes))
    y_one_hot.scatter_(1, torch.unsqueeze(torch.tensor(y), 1), 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.1, random_state=42)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    print("x_train\n", x_train[:5])
    print("y_train\n", y_train[:5])
    print("x_test\n", x_test[:5])
    print("y_test\n", y_test[:5])

    torch.save(species, "./~data/species.pt")
    torch.save(x_train, "./~data/x_train.pt")
    torch.save(x_test, "./~data/x_test.pt")
    torch.save(y_train, "./~data/y_train.pt")
    torch.save(y_test, "./~data/y_test.pt")


def data_load_species():
    return torch.load("./~data/species.pt")


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
