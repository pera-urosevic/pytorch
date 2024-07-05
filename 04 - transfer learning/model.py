import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
from data import dataset
from config import device
from performance import Performance


def model_train(images_dir):
    batch_size = 64
    learning_rate = 0.001
    min_epochs = 5
    max_epochs = 100

    train_data, test_data, classes_map = dataset(images_dir)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    num_classes = len(classes_map)

    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    performance = Performance(name="resnet",
                              collection=images_dir,
                              min_epochs=min_epochs,
                              max_epochs=max_epochs,
                              patience=5,
                              delta=0.1)

    model.to(device)
    for epoch in range(max_epochs):
        model.train()
        count = 0
        loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            count += 1
            if i % 10 == 0 and i != 0:
                print(f'[{epoch + 1}, {i:4d}] loss: {loss / count:.3f}')

        accuracy = model_accuracy(model, test_loader)
        print(f'[{epoch + 1}] loss: {loss / count:.3f}, accuracy: {accuracy:.3f}')

        performance.on_epoch(model, epoch, loss, accuracy)
        if performance.is_done():
            print(
                f"Performance early stop, best Epoch [{performance.best_epoch}], Metric: {performance.best_accuracy:.4f}"
            )
            model.load_state_dict(performance.best_state)
            break

    torch.save(model, "./~model/resnet.pt")


def model_accuracy(model, test_loader):
    with torch.inference_mode():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total


if __name__ == "__main__":
    model_train('food_small')
