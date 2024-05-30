from _model import model_load
from _eval import eval_model, eval_model_plot

model = model_load()
eval_model(model)
eval_model_plot(model)
