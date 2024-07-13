import copy
import mlflow
import torch.nn as nn
import torch.optim as optim
from _data import (
    data_load_x_train,
    data_load_y_train,
    data_load_x_test,
    data_load_y_test,
)
from _model import Model, model_accuracy


def train_model(num_epochs, early_stoping_patience=20):
    x_test = data_load_x_test()
    y_test = data_load_y_test()
    x_train = data_load_x_train()
    y_train = data_load_y_train()
    model = Model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0
    best_model_weights = None
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.data.get
    with mlflow.start_run():
        for epoch in range(num_epochs):
            model.train()

            optimizer.zero_grad()

            outputs = model(x_train)
            loss = criterion(outputs, y_train)

            loss.backward()
            optimizer.step()

            accuracy = model_accuracy(model, x_test, y_test)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = early_stoping_patience
            else:
                patience -= 1
                if patience == 0:
                    break

    model.load_state_dict(best_model_weights)
    return model, best_accuracy
