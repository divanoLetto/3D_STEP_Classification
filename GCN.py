import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class AttentionModule(torch.nn.Module):
    """
    Attention Module to make a pass on graph.
    """
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.setup_weights(dim)
        self.init_parameters()

    def setup_weights(self, dim):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(dim, dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class GCN(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 16
        self.bottle_neck_neurons = 8

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v2(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v2, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 64
        self.filters_3 = 32
        self.filters_4 = 32
        self.bottle_neck_neurons = 16

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.convolution_4 = GCNConv(in_channels=self.filters_3, out_channels=self.filters_4)
        self.attention = AttentionModule(self.filters_4)
        self.fully_connected_first = nn.Linear(self.filters_4, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_4(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                         p=self.dropout,
                                         training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v3(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v3, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 16
        self.bottle_neck_neurons = 16

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v4(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v4, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 32
        self.bottle_neck_neurons = 32

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v5(nn.Module):
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v5, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 80
        self.filters_2 = 64
        self.filters_3 = 64
        self.bottle_neck_neurons_1 = 64
        self.bottle_neck_neurons_2 = 32

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons_1)
        self.fully_connected_second = nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons_2, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(x=features, edge_index=adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = nn.functional.relu(self.fully_connected_second(scores))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score



class GCN_FC_v2(nn.Module):
    """
    without the second FC.
    """
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_FC_v2, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 16
        self.bottle_neck_neurons = 8

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.attention = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.num_classes)
        # self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = self.attention(features)
        pooled_features = torch.t(pooled_features)

        scores = self.fully_connected_first(pooled_features)
        # scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_att_v2(nn.Module):
    """
    Instead of th attention mechanism, an unweighted sum is performed.
    """
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_att_v2, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 16
        self.bottle_neck_neurons = 8

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        # self.unweighted_sum = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = torch.mean(features, dim=0).unsqueeze(0) # sum(features).unsqueeze(0)
        # pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_att_v3(nn.Module):
    """
    Instead of th attention mechanism, an degree weighted sum is performed.
    """
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_att_v3, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 16
        self.bottle_neck_neurons = 8

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features, neighbors):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        degrees_tmp = np.array([len(l) for l in neighbors])
        degrees = torch.from_numpy(degrees_tmp).to(device)  # torch.from_numpy(degrees_tmp/sum(degrees_tmp)).to(device)
        pooled_features = torch.mean(torch.mul(features,degrees[:,None]), 0)
        # pooled_features = np.average(features.detach().cpu(), axis=0, weights=degrees, returned=False)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pooled_features = torch.from_numpy(pooled_features).to(device)
        # pooled_features.requires_grad = True
        pooled_features = pooled_features.unsqueeze(0)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features.float()))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v4_att_v2(nn.Module):
    """
    Instead of th attention mechanism, an unweighted avg is performed.
    """
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v4_att_v2, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 32
        self.bottle_neck_neurons = 32

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        # self.unweighted_sum = AttentionModule(self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        pooled_features = torch.mean(features, dim=0).unsqueeze(0) # sum(features).unsqueeze(0)
        # pooled_features = torch.t(pooled_features)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


class GCN_CN_v4_att_v3(nn.Module):
    """
    Instead of th attention mechanism, an degree weighted sum is performed.
    """
    def __init__(self, feature_dim_size, num_classes, dropout):
        super(GCN_CN_v4_att_v3, self).__init__()

        self.number_labels = feature_dim_size
        self.num_classes = num_classes

        self.filters_1 = 64
        self.filters_2 = 32
        self.filters_3 = 32
        self.bottle_neck_neurons = 32

        self.convolution_1 = GCNConv(in_channels=self.number_labels, out_channels=self.filters_1)
        self.convolution_2 = GCNConv(in_channels=self.filters_1, out_channels=self.filters_2)
        self.convolution_3 = GCNConv(in_channels=self.filters_2, out_channels=self.filters_3)
        self.fully_connected_first = nn.Linear(self.filters_3, self.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.bottle_neck_neurons, self.num_classes)
        self.dropout = dropout

    def forward(self, adj, features, neighbors):
        features = self.convolution_1(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_3(features, adj)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        degrees_tmp = np.array([len(l) for l in neighbors])
        degrees = torch.from_numpy(degrees_tmp).to(device)  # torch.from_numpy(degrees_tmp/sum(degrees_tmp)).to(device)
        pooled_features = torch.mean(torch.mul(features,degrees[:,None]), 0)
        # pooled_features = np.average(features.detach().cpu(), axis=0, weights=degrees, returned=False)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # pooled_features = torch.from_numpy(pooled_features).to(device)
        # pooled_features.requires_grad = True
        pooled_features = pooled_features.unsqueeze(0)

        scores = nn.functional.relu(self.fully_connected_first(pooled_features.float()))
        scores = self.scoring_layer(scores)
        score = F.log_softmax(scores, dim=1)
        return score


