from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_labels = true_labels.type(torch.int64)
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

    return Adj_block_idx_row, Adj_block_idx_cl


def get_graphpool(batch_graph, device):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)


def get_batch_data(batch_graph, device):
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    X_concat = torch.from_numpy(X_concat).to(device)
    # graph-level sum pooling

    adjj = np.concatenate([graph.edge_mat for graph in batch_graph], 0)
    adjj = torch.from_numpy(adjj).to(device)

    graph_labels = np.array([graph.label for graph in batch_graph])
    graph_labels = torch.from_numpy(graph_labels).to(device)

    return X_concat, graph_labels, adjj.to(torch.int64)


def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def train(mmodel, optimizer, train_graphs, batch_size, num_classes, device):
    # Turn on the train mode
    mmodel.train()
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), batch_size):
        end = start + batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        # load graph batch
        X_concat, graph_labels, adjj = get_batch_data(batch_graph, device=device)
        graph_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        # model probability scores
        prediction_scores = mmodel(adjj, X_concat)

        loss = cross_entropy(prediction_scores, graph_labels)
        # backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmodel.parameters(), 0.5)  # prevent the exploding gradient problem
        optimizer.step()


def evaluate(mmodel, current_graphs, batch_size, num_classes, device, out_dir, last_round=False):
    # Turn on the evaluation mode
    mmodel.eval()
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(current_graphs))
        for i in range(0, len(current_graphs), batch_size):
            sampled_idx = idx[i:i + batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [current_graphs[j] for j in sampled_idx]
            # load graph batch
            test_X_concat, test_graph_labels, test_adj = get_batch_data(batch_test_graphs, device=device)
            # model probability scores
            prediction_scores = mmodel(test_adj, test_X_concat)

            test_graph_labels = label_smoothing(test_graph_labels, num_classes)
            loss = cross_entropy(prediction_scores, test_graph_labels)
            total_loss += loss.item()
            prediction_output.append(prediction_scores.detach())

    # model probabilities output
    prediction_output = torch.cat(prediction_output, 0)
    # predicted labels
    predictions = prediction_output.max(1, keepdim=True)[1]
    # real labels
    labels = torch.LongTensor([graph.label for graph in current_graphs]).to(device)
    # num correct predictions
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    accuracy = correct / float(len(current_graphs))

    # confusion matrix and class accuracy
    matrix = confusion_matrix(np.array(labels.cpu()), np.array(predictions.cpu()))
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    acc_x_class = matrix.diagonal() * 100

    if last_round:
        # plot and save statistics
        print("Accuracy per class :")
        print(acc_x_class)
        with open(out_dir + "/test_results.txt", 'w') as f:
            f.write("Evaluate: loss on test: "+ str(total_loss/len(current_graphs)) + " and accuracy: " + str(accuracy * 100)+"\n")
            f.write("Accuracy per class : "+ str(matrix.diagonal())+"\n")
            f.write(metrics.classification_report(np.array(labels.cpu()), np.array(predictions.cpu()), digits=3))

        ax = sns.heatmap(matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.savefig(out_dir + "/Confusion Matrix")

    return total_loss/len(current_graphs), accuracy, acc_x_class