import copy

import numpy as np
from sklearn.metrics import log_loss
import torch


class Performance:
    def __init__(self, name, collection, min_epochs, max_epochs, patience=5, delta=0):
        self.name = name
        self.collection = collection
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.log_loss = np.empty(max_epochs)
        self.log_accuracy = np.empty(max_epochs)
        self.best_accuracy = None
        self.best_state = None
        self.best_epoch = None
        self.done = False

    def on_epoch(self, model, epoch, loss, accuracy):
        self.log_loss[epoch - 1] = loss
        self.log_accuracy[epoch - 1] = accuracy
        if self.is_improving(accuracy):
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_accuracy = accuracy
            self.counter = 0
        else:
            self.counter += 1
            self.done = (self.counter >= self.patience and epoch >= self.min_epochs) or (epoch >= self.max_epochs)
            if self.done:
                self.log_loss.resize(epoch)
                self.log_accuracy.resize(epoch)
                self.write_log()

    def is_improving(self, accuracy):
        return self.best_accuracy is None or (accuracy - self.best_accuracy) > self.delta

    def is_done(self):
        return self.done

    def write_log(self):
        data = {"loss": self.log_loss, "accuracy": self.log_accuracy}
        torch.save(data, f"~train/{self.name}-{self.collection}.pt")
