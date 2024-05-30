from _model import model_save
from _tune import tune_hyperparams_load
from _train import train_model

hyperparams = tune_hyperparams_load()
model, accuracy = train_model(num_features=hyperparams["num_features"], num_epochs=1000)
model_save(model)
