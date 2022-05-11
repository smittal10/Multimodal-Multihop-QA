
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv

 
# class ToyNet(nn.Module):
#     def __init__(self):
#         super(ToyNet, self).__init__()
#         num_features = 3584
#         num_classes = 1
#         self.conv1 = GCNConv(num_features, 2048)
#         self.conv2 = GCNConv(2048, 512)
#         self.conv3 = GCNConv(512, 256)
#         self.conv4 = GCNConv(256, 64)
#         self.conv5 = GCNConv(64, int(num_classes))
#         self._relu = nn.ReLU()
#         self._sigmoid = nn.Sigmoid()
#         self._dropout = nn.Dropout(0.3)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv2(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv3(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv4(x, edge_index)
#         x = self._relu(x)
#         x = self._dropout(x)
#         x = self.conv5(x, edge_index)

        
#         return self._sigmoid(x)


embed_dim = 3584
import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.dense import Linear

class ToyNet(torch.nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = SAGEConv((-1, -1), 2048)
        self.conv2 = SAGEConv((-1, -1), 1024)
        self.conv3 = SAGEConv((-1, -1), 512)
        self.conv4 = SAGEConv((-1, -1), 256)
        self.conv5 = SAGEConv((-1, -1), 256)
        self.conv6 = SAGEConv((-1, -1), 128)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 2)

        self.blinq = torch.nn.BatchNorm1d(512)
        self.blinc = torch.nn.BatchNorm1d(512)
        self.blini = torch.nn.BatchNorm1d(512)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()  
        self._softmax = torch.nn.Softmax(dim=-1)      
        print("Just want to be sure")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = self.lin1(x)
        x = self.act1(x)
        x_out = self.lin2(x)
        x = self.act2(x_out)      
        x = F.dropout(x, p=0.5, training=self.training)
        x_log = self.lin3(x)
        x = self._softmax(x_log)
        # return (x_log, x , x_out)
        return x_log