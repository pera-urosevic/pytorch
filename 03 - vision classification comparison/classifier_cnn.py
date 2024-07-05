import torch.nn as nn
import torch.optim as optim

from data import output_shape
from config import device
from classifier import BaseClassifier
from performance import Performance


class CNNModel(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, output_shape)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNClassifier(BaseClassifier):
    def __init__(self, collection):
        model = CNNModel(output_shape).to(device)
        super().__init__("cnn", model, collection)

    def train(self):
        min_epochs = 20
        max_epochs = 200
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=0.001)
        performance = Performance(self.name, self.collection, min_epochs, max_epochs, patience=10, delta=0.1)
        self.model = super().train(
            criterion,
            optimizer,
            performance,
            num_epochs=max_epochs,
            epoch_check=1,
        )
        self.save()


if __name__ == "__main__":
    # cnn = CNNClassifier("mnist")
    # cnn.train()
    cnn = CNNClassifier("simpsons-color")
    cnn.train()
    cnn = CNNClassifier("simpsons-mono")
    cnn.train()
    pass
