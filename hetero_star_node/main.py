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
data_root =  "/home/ubuntu/webqa_data"
batch_size = 64
epochs = 500
device = "cuda"
save_every = 1
print_step = 270
val_step = 500

webqa_dataset_train = WebQnaDataset(data_root)
webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size, shuffle=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, 1, shuffle=False)
# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)

toy_model = ToyNet()

graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','rlink1','ques'), ('img_src','rlink2','ques'), ('txt_src','ti_link','img_src'),
('txt_src','tt_link','txt_src'),('img_src','ii_link','img_src'),'img_src','it_link','txt_src'])

#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
toy_model = to_hetero(toy_model, graph_meta)
toy_model = toy_model.to(device)

# criterion = torch.nn.BCELoss()
class_weights = torch.tensor([1,10], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights)
#criterion = FocalLoss(gamma=1, alpha=0.9, reduction='mean')
optimizer = torch.optim.AdamW(toy_model.parameters(), lr=0.00002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
toy_model.train()
 
g_ctr = 0
best_f1s = 0
for epoch in range(epochs):
    g_ctr = 0
    for idx, datum in enumerate(webqa_dataloader_train):
        g_ctr += 1
        toy_model.train()
        optimizer.zero_grad()
        
        datum = datum.to(device)
        #print(datum)
        #assert(False)
        #outp, pred = toy_model(datum.x_dict, datum.edge_index_dict)
        outp = toy_model(datum.x_dict, datum.edge_index_dict)
        #print(outp)
        #print(datum)
        # outp = outp.squeeze(1)
        # print(datum.y.shape, outp.shape)
        #loss = criterion(outp, datum.y_dict)
        loss_img = criterion(outp['img_src'], datum.y_dict['img_src'])
        loss_txt = criterion(outp['txt_src'], datum.y_dict['txt_src'])
        loss = loss_img + loss_txt
        loss.backward()
        optimizer.step()
        if g_ctr % print_step == 0:
            #print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
            print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), optimizer.param_groups[0]['lr']))
        
        if g_ctr % val_step == 0:
            toy_model.eval()
            val_pred = 0.
            total_vals = 0.
            all_gt = []
            all_pred = []
            save_gt_ = []
            save_pred_ = []
            for idx, datum_val in enumerate(webqa_dataloader_val):
                #total_vals += datum_val.x.shape[0]
                total_vals += datum_val.x_dict['img_src'].shape[0]
                total_vals += datum_val.x_dict['txt_src'].shape[0]
                datum_val = datum_val.to(device)
                pred = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
                loss_img_val = criterion(pred['img_src'], datum_val.y_dict['img_src'])
                loss_txt_val = criterion(pred['txt_src'], datum_val.y_dict['txt_src'])
                loss_val = loss_img_val + loss_txt_val
                # outp = outp.squeeze(1)
                # outp = outp >0.5
                # print(pred.shape)
                #pred = torch.argmax(pred, dim=-1)
                pred_img = torch.argmax(pred['img_src'], dim=-1)
                pred_txt = torch.argmax(pred['txt_src'], dim=-1)
                save_pred_.extend(list(pred_img.detach().cpu().numpy()))
                save_pred_.extend(list(pred_txt.detach().cpu().numpy()))
                
                # gt = torch.argmax(datum_val.y, dim=-1)
                #gt = datum_val.y
                gt_img = datum_val.y_dict['img_src']
                gt_txt = datum_val.y_dict['txt_src']
                save_gt_.extend(list(gt_img.detach().cpu().numpy()))
                save_gt_.extend(list(gt_txt.detach().cpu().numpy()))
                
                # print(datum_val.x.shape[0],torch.sum(outp==0))
                # val_pred += torch.sum(outp==datum_val.y).detach().cpu().item()
                #val_pred += torch.sum(pred==gt).detach().cpu().item()
                val_pred_img = torch.sum(pred_img==gt_img).detach().cpu().item()
                val_pred_txt = torch.sum(pred_txt==gt_txt).detach().cpu().item()
                val_pred += val_pred_img + val_pred_txt
                # all_gt.extend(datum_val.y.detach().cpu())
                #all_gt.extend(gt.detach().cpu())
                #all_pred.extend(pred.detach().cpu())
                
                all_gt.extend(gt_img.detach().cpu())
                all_gt.extend(gt_txt.detach().cpu())
                all_pred.extend(pred_img.detach().cpu())
                all_pred.extend(pred_txt.detach().cpu())
            f1s = f1_score(all_gt,all_pred)
            if f1s > best_f1s:
                # save_preds
                best_f1s = f1s
                print("Saving the results, F1 = {f1s}")
                df = pd.DataFrame()
                df["Pred"] = save_pred_
                df["GT"] = save_gt_
                df.to_csv("Predictions.csv", index=False) 
            print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr, (val_pred/total_vals)*100), f1s)
    #scheduler.step(loss_val)
    scheduler.step()
    if f1s > best_f1s:
        best_f1s = f1s
        torch.save({'epoch': epoch,
            'state_dict': toy_model.state_dict()},
            'train.ckpt')