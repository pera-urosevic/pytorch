import itertools
from matplotlib import pyplot as plt
import pandas as pd


def rectangular_layout(num):
    sqrt_num = int(num**0.5)
    for i in range(sqrt_num, 0, -1):
        if num % i == 0:
            return (i, num // i)
    return None


def analyze_plot():
    data = pd.read_csv("~data/iris.csv")
    feature_names = data.columns[:-1]
    pairs = list(itertools.combinations(feature_names, 2))
    pairs_count = len(pairs)
    label_name = data.columns[-1:][0]
    labels = data[label_name].unique()
    layout = rectangular_layout(pairs_count)

    fig, axes = plt.subplots(nrows=layout[0], ncols=layout[1], figsize=(20 // layout[0], 20 // layout[1]))
    axes_flat = list(itertools.chain.from_iterable(axes))
    fig.tight_layout()
    colormap = plt.get_cmap(plt.colormaps["Dark2"])

    for i, columns in enumerate(pairs):
        axis = axes_flat[i]
        for c, l in enumerate(labels):
            x = data[data[label_name] == l][columns[0]]
            y = data[data[label_name] == l][columns[1]]
            axis.scatter(x, y, color=colormap.colors[c])
            axis.set(xlabel=columns[0], ylabel=columns[1])

    fig.legend(labels=labels, loc="upper right")
    plt.show()
