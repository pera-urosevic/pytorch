import datetime
import torch
import optuna
from optuna.trial import TrialState
from _data import (
    data_load_x_train,
    data_load_y_train,
    data_load_x_test,
    data_load_y_test,
)
from _model import model_accuracy
from _train import train_model


def tune_model():
    x_test = data_load_x_test()
    y_test = data_load_y_test()

    def objective(trial):
        num_features = trial.suggest_int("num_features", 10, 1000)
        model, loss = train_model(num_features=num_features, num_epochs=100)
        test_accuracy = model_accuracy(model, x_test, y_test)
        return test_accuracy

    study = optuna.create_study(
        study_name=datetime.datetime.now().isoformat(),
        load_if_exists=False,
        direction="maximize",
        storage="sqlite:///~tune/study.db",
    )
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    torch.save(study.best_params, "./~tune/hyperparams.pt")


def tune_hyperparams_load():
    return torch.load("./~tune/hyperparams.pt")
