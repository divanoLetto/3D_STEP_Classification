import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle
import random


def plot_precision_recall(y_truth, y_score, save_path):
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_score.shape[1]
    Y_test = np.empty(shape=y_score.shape, dtype="int64")
    for example, c in enumerate(y_truth):
        Y_test[example, :] = 0
        Y_test[example, int(c)] = 1
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true=Y_test[:, i], probas_pred=y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")
    plt.savefig(save_path + '/Micro-averaged-precision.png')

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.savefig(save_path + '/Precision-Recall-multi-class.png')
    plt.show()


def plot_training_flow(ys,names, y_axis, path, fig_name):
    epochs = range(len(ys[0]))
    plt.xlabel('Epoch')
    for i, y in enumerate(ys):
        plt.plot(epochs, y, label=names[i])
        plt.legend()
        plt.ylabel(y_axis)
    plt.savefig(path + fig_name)
    plt.clf()


def split_data(all_set, perc):
    training_set = []
    test_set = []
    graphs_x_class = {}
    for g in all_set:
        if g.label not in graphs_x_class.keys():
            graphs_x_class[g.label] = []
        graphs_x_class[g.label].append(g)

    for c, graphs in graphs_x_class.items():
        num_elem = len(graphs)
        num_train_elem = int(num_elem * perc)
        num_test_elem = num_elem - num_train_elem
        random.shuffle(graphs)
        training_set.extend(graphs[:num_train_elem])
        test_set.extend(graphs[-num_test_elem:])

    return training_set, test_set


def print_data_commposition(set):
    dd = {}
    for g in set:
        if g.label not in dd.keys():
            dd[g.label] = []
        dd[g.label].append(g)
    for cl in dd.keys():
        print("class:", str(cl), " - num elements:", len(dd[cl]), " - elements: ", [f.name_graph for f in dd[cl]])