import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.dense import Linear
# from .clip_model import build_model as build_clip_model

graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','link3','ques'), ('img_src','link4','ques')])

class JointModel(torch.nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        # self.clip = build_clip_model(state_dict)
        # self.clip = self.clip.visual()
        self.gnn = ToyNet()
        self.gnn = to_hetero(self.gnn, graph_meta)

    def forward(self, clip_model, datum):
        # use vector in  '[CLS]' position as the node features
        images = datum.x_dict['img_src']
        node_feats = clip_model(images.half())
        # set node_feats as the initial node features in GNN
        datum['img_src'].x = torch.cat((datum.x_dict['img_src_cap'], node_feats),dim=-1)
        ##concatenate with image captions TODO
        out = self.gnn(datum.x_dict, datum.edge_index_dict)
        return out


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