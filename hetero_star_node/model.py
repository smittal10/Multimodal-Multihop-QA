
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

 
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
class EdgeClass(nn.Module):  
    def __init__(self, edgetypes, node_dim):
        super(EdgeClass, self).__init__()

        #FFN for each edge type
        self.node_dim= node_dim
        self.edge_layers={}
        self.hidden_dim_edge= 256

        for type in edgetypes:
            layername= type
            layer= nn.ModuleList([nn.Linear(2*self.node_dim ,self.hidden_dim_edge), nn.Linear(self.hidden_dim_edge,2)])
            layer = layer.to("cuda:0")
            self.edge_layers[layername]= layer
        
    
    def forward(self, src, tgt, edge_type):
        module_list= self.edge_layers[edge_type]
        edge_feat = torch.cat([src,tgt], dim=1)
        for num,layer in enumerate(module_list): 
            
            edge_feat= layer(edge_feat)
            if(num==len(module_list)-1):
                break
            edge_feat= F.relu(edge_feat)
        return edge_feat    


class ToyNet(torch.nn.Module):
    def __init__(self,):
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
    

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x_hdim = x
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        x_log = self.lin3(x)
        x = self._softmax(x_log)
        #pred_edges= self.getEdgePreds(x,edge_index)
        return x_log, x_hdim
class SuperNet(nn.Module):
    def __init__(self, edge_types_preds, device="cuda:0",):
        super(SuperNet, self).__init__()
        self.GAT= ToyNet()
        self.device=device
        self.gat_output_dim= 128
        self.edge_types= edge_types_preds
        self.edgeclass_head= EdgeClass(self.edge_types, self.gat_output_dim)
        self.edgeclass_head= self.edgeclass_head.to(device)
    def getEdgePreds(self,x,edge_index):
        edges_preds={}
        
        for edge_type in self.edge_types:
            src_idx,tgt_idx = edge_index[edge_type]
            src_type,link_type,tgt_type= edge_type
            src_feat= x[src_type][src_idx]
            tgt_feat= x[tgt_type][tgt_idx]
            edges_preds[edge_type]= self.edgeclass_head(src_feat,tgt_feat,edge_type)
        return edges_preds

    def forward(self, x, edge_index):
        x, x_hdim= self.GAT(x, edge_index)
      
        pred_edges= self.getEdgePreds(x_hdim,edge_index)
        return x, pred_edges


    