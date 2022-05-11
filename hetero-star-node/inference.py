from numpy.core.numeric import outer
from scipy.sparse import data
import torch
from dataset import WebQnaDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model import ToyNet
from sklearn.metrics import f1_score
from focalloss import FocalLoss
#from torch_geometric.nn import to_hetero
import torch
from torch.nn import ReLU
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero
data_root = "/home/ubuntu/WebQna/nodes-2611"
batch_size = 64
epochs = 500
device = "cuda"
save_every = 1
print_step = 270
val_step = 500

webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size, shuffle=False)
# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)

toy_model = ToyNet()

# model = Sequential('x, edge_index', [
#     (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (Linear(-1, 2), 'x -> x'),
# ])
# toy_model = model
graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','link3','ques'), ('img_src','link4','ques')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
toy_model = to_hetero(toy_model, graph_meta)
toy_model = torch.load("best_node_based.ckpt",map_location="cpu")
toy_model = toy_model.to(device)


toy_model.eval()
val_pred = 0.
total_vals = 0.
all_gt = []
all_pred = []
node_features = []
logits = []
for idx, datum_val in enumerate(webqa_dataloader_val):
    #total_vals += datum_val.x.shape[0]
    total_vals += datum_val.x_dict['img_src'].shape[0]
    total_vals += datum_val.x_dict['txt_src'].shape[0]
    datum_val = datum_val.to(device)
    with torch.no_grad():
        pred = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
        logits.append(F.softmax(pred['txt_src'],dim=-1))
        logits.append(F.softmax(pred['img_src'],dim=-1))
        # node_features.append(['txt_src'])
        # node_features.append(['img_src'])
        
        # outp = outp.squeeze(1)
        # outp = outp >0.5
        # print(pred.shape)
        #pred = torch.argmax(pred, dim=-1)
        pred_img = torch.argmax(pred['img_src'], dim=-1)
        pred_txt = torch.argmax(pred['txt_src'], dim=-1)
        
        # gt = torch.argmax(datum_val.y, dim=-1)
        #gt = datum_val.y
        gt_img = datum_val.y_dict['img_src']
        gt_txt = datum_val.y_dict['txt_src']
        # print(datum_val.x.shape[0],torch.sum(outp==0))
        # val_pred += torch.sum(outp==datum_val.y).detach().cpu().item()
        #val_pred += torch.sum(pred==gt).detach().cpu().item()
        val_pred_img = torch.sum(pred_img==gt_img).detach().cpu().item()
        val_pred_txt = torch.sum(pred_txt==gt_txt).detach().cpu().item()
        val_pred += val_pred_img + val_pred_txt
        # all_gt.extend(datum_val.y.detach().cpu())
        #all_gt.extend(gt.detach().cpu())
        #all_pred.extend(pred.detach().cpu())
        
        all_gt.extend(gt_txt.detach().cpu())
        all_gt.extend(gt_img.detach().cpu())
        
        all_pred.extend(pred_txt.detach().cpu())
        all_pred.extend(pred_img.detach().cpu())
                
f1s = f1_score(all_gt,all_pred)
# torch.save(all_pred,"preds_from_node_based.pt")
torch.save(logits,"logits_val.pt")
# torch.save(node_features,"node_features.pt")
print("Val Acc :: {}  F1 Score :: ".format( (val_pred/total_vals)*100), f1s)
