import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


def data_generate():
    x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    torch.save(x, "./~data/x.pt")
    torch.save(y, "./~data/y.pt")
    torch.save(x_train, "./~data/x_train.pt")
    torch.save(x_test, "./~data/x_test.pt")
    torch.save(y_train, "./~data/y_train.pt")
    torch.save(y_test, "./~data/y_test.pt")


if __name__ == "__main__":
    data_generate()
