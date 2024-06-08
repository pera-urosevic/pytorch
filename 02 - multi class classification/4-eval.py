from _model import model_load
from _eval import eval_model, eval_model_accuracy, eval_model_plot, eval_model_unseen
from _data import (
    data_load_species,
    data_load_x_test,
    data_load_y_test,
)

species = data_load_species()
x_test = data_load_x_test()
y_test = data_load_y_test()

model = model_load()
model.eval()

test_predicted, test_labels = eval_model(model, x_test, y_test)

# eval_model_accuracy(test_predicted, test_labels)

eval_model_plot(species, test_predicted, test_labels)

# eval_model_unseen(model, torch.tensor([4.0, 3.3, 1.7, 0.5]), "Iris-setosa")
