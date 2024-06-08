import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, 3)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def model_predict(model, x, y):
    with torch.no_grad():
        p = model(x)
        _, predicted = torch.max(p, 1)
        _, labels = torch.max(y, 1)
        return predicted, labels


def model_accuracy_score(predicted, labels):
    correct = (predicted == labels).sum().item()
    accuracy_score = correct / len(labels)
    return accuracy_score


def model_accuracy(model, x, y):
    predicted, labels = model_predict(model, x, y)
    accuracy_score = model_accuracy_score(predicted, labels)
    return accuracy_score


def model_save(model):
    torch.save(model, "./~model/model.pt")


def model_load():
    model = torch.load("./~model/model.pt")
    model.eval()
    return model
