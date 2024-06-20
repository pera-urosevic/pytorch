import torch.nn as nn
import torch.optim as optim
from config import device
from data import input_shape, output_shape
from classifier import BaseClassifier
from performance import Performance


class SimpleModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear1 = nn.Linear(input_shape, hidden_units)
        self.linear2 = nn.Linear(hidden_units, output_shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class SimpleClassifier(BaseClassifier):
    def __init__(self, collection):
        model = SimpleModel(input_shape, 128, output_shape).to(device)
        super().__init__("simple", model, collection)

    def train(self):
        min_epochs = 10
        max_epochs = 100
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=self.model.parameters(), lr=0.01)
        performance = Performance(self.name, self.collection, min_epochs, max_epochs, patience=3, delta=0.3)
        self.model = super().train(
            criterion,
            optimizer,
            performance,
            num_epochs=max_epochs,
            epoch_check=1,
        )
        self.save()


if __name__ == "__main__":
    # simple = SimpleClassifier("mnist")
    # simple.train()
    # simple = SimpleClassifier("simpsons-color")
    # simple.train()
    # simple = SimpleClassifier("simpsons-mono")
    # simple.train()
    pass
