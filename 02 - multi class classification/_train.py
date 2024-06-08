import copy
import torch
import torch.nn as nn
import torch.optim as optim
from _data import (
    data_load_x_train,
    data_load_y_train,
    data_load_x_test,
    data_load_y_test,
)
from _model import Model, model_accuracy


class Performance:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_metric = None
        self.best_state = None
        self.best_epoch = None

    def check(self, model, epoch, metric):
        if self.best_metric is None or (metric - self.best_metric) > self.delta:
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_metric = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_model(num_features, num_epochs=100, epoch_check=10, patience=10, delta=0.1):
    x_test = data_load_x_test()
    y_test = data_load_y_test()
    x_train = data_load_x_train()
    y_train = data_load_y_train()
    model = Model(num_features=num_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    performance = Performance(patience=patience, delta=delta)

    for epoch in range(1, num_epochs + 1):
        model.train()

        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, torch.argmax(y_train, dim=1))

        loss.backward()
        optimizer.step()

        accuracy = model_accuracy(model, x_test, y_test)

        if epoch % epoch_check == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        if performance.check(model, epoch, accuracy):
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
            break

    model.load_state_dict(performance.best_state)
    return model, performance.best_metric
