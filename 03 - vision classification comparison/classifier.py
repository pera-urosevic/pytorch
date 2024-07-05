import torch
from config import test, device
from data import data_loader


class BaseClassifier:
    def __init__(self, name, model, collection):
        self.name = name
        self.model = model
        self.collection = collection
        self.train_loader = data_loader(collection, "train")
        self.test_loader = data_loader(collection, "test")

    def model_accuracy(self):
        with torch.inference_mode():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100 * correct / total

    def train(self, criterion, optimizer, performance, num_epochs=100, epoch_check=10):
        print()
        print(f"{self.name} / {self.collection}")
        num_epochs = 2 if test else num_epochs
        for epoch in range(1, num_epochs + 1):
            loss = 0
            for images, labels in self.train_loader:
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(images.to(device))
                batch_loss = criterion(outputs, labels.to(device))
                loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

            accuracy = self.model_accuracy()
            if epoch % epoch_check == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            performance.on_epoch(self.model, epoch, loss, accuracy)
            if performance.is_done():
                print(
                    f"Performance early stop, best Epoch [{performance.best_epoch}], Metric: {performance.best_accuracy:.4f}"
                )
                break

        self.model.load_state_dict(performance.best_state)
        performance.write_log()

    def save(self):
        torch.save(self.model, f"./~model/{self.name}-{self.collection}.pt")

    def load(self):
        self.model = torch.load(f"./~model/{self.name}-{self.collection}.pt")
        self.model.eval()
