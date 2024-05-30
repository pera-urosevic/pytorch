import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def model_accuracy(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        predicted = outputs.round()
        accuracy = (predicted.eq(y).sum() / float(y.shape[0])).item()
    return accuracy


def model_save(model):
    torch.save(model, "./~model/model.pt")


def model_load():
    model = torch.load("./~model/model.pt")
    model.eval()
    return model
