from numpy.core.numeric import outer
from scipy.sparse import data
import torch
import clip
from dataset_raw_imgs import WebQnaDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model_joint import JointModel
from sklearn.metrics import f1_score
from focalloss import FocalLoss
#from torch_geometric.nn import to_hetero
from torch.nn import ReLU
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
data_root = "/home/ubuntu/WebQna/nodes-2611"
batch_size =16
epochs = 500
device = "cuda"
save_every = 1
print_step = 240
val_step = 500

webqa_dataset_train = WebQnaDataset(data_root)
webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size=batch_size, shuffle=True,num_workers=12)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size=batch_size, shuffle=False,num_workers=12)
# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
clip_model = clip_model.visual
clip.model.convert_weights(clip_model)
clip_model.to(device)

toy_model = JointModel()

# toy_model = model
graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','link3','ques'), ('img_src','link4','ques')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
toy_model.gnn = torch.load("best_node_based_clip.ckpt",map_location="cpu")
toy_model.to(device)

# criterion = torch.nn.BCELoss()
class_weights = torch.tensor([1,10], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights)
#criterion = FocalLoss(gamma=1, alpha=0.9, reduction='mean')
# optimizer = torch.optim.AdamW(toy_model.parameters(), lr=0.00002)
optimizer = torch.optim.Adam([{'params': clip_model.parameters(),'lr': 5e-6},{'params': toy_model.gnn.parameters()}], lr=1e-5,eps=1e-6,weight_decay=0.2)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
toy_model.train()
 
g_ctr = 0
best_f1s = 0
local_batch_size = batch_size
total_batch_size = 64
grad_acc_steps = total_batch_size / local_batch_size
for epoch in range(epochs):
    g_ctr = 0
    for idx, datum in enumerate(webqa_dataloader_train):
        g_ctr += 1
        # clip_model.train()
        toy_model.train()
        # optimizer.zero_grad()
        
        datum = datum.to(device)
        #print(datum)
        #assert(False)
        #outp, pred = toy_model(datum.x_dict, datum.edge_index_dict)
        outp = toy_model(clip_model, datum)
        #print(outp)
        #print(datum)
        # outp = outp.squeeze(1)
        # print(datum.y.shape, outp.shape)
        #loss = criterion(outp, datum.y_dict)
        loss_img = criterion(outp['img_src'], datum.y_dict['img_src'])
        loss_txt = criterion(outp['txt_src'], datum.y_dict['txt_src'])
        loss = (loss_img + loss_txt) / grad_acc_steps
        loss.backward()
        if (idx+1)%grad_acc_steps == 0:
            convert_models_to_fp32(clip_model)
            optimizer.step()
            optimizer.zero_grad()
            clip.model.convert_weights(clip_model)
        if g_ctr % print_step == 0:
            #print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
            print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), optimizer.param_groups[0]['lr']))
        
    if (epoch+1) % 1 == 0:
        clip_model.eval()
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
            with torch.no_grad():
                pred = toy_model(clip_model,datum_val)
            # loss_img_val = criterion(pred['img_src'], datum_val.y_dict['img_src'])
            # loss_txt_val = criterion(pred['txt_src'], datum_val.y_dict['txt_src'])
            # loss_val = loss_img_val + loss_txt_val
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
            
            all_gt.extend(gt_txt.detach().cpu())
            all_gt.extend(gt_img.detach().cpu())
            
            all_pred.extend(pred_txt.detach().cpu())
            all_pred.extend(pred_img.detach().cpu())
            
        f1s = f1_score(all_gt,all_pred)
        if f1s > best_f1s:
            # save_preds
            torch.save(clip_model,"fine_tuned.ckpt")
            torch.save(toy_model.gnn,"best_joint_gnn.ckpt")
            best_f1s = f1s
            # print("Saving the results, F1 = {f1s}")
            # df = pd.DataFrame()
            # df["Pred"] = save_pred_
            # df["GT"] = save_gt_
            # df.to_csv("Predictions.csv", index=False)
            # torch.save(all_pred,"preds_from_node_based.pt")
        print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr, (val_pred/total_vals)*100), f1s)
    #scheduler.step(loss_val)
    scheduler.step()
    # if f1s > best_f1s:
    #     best_f1s = f1s
    #     torch.save({'epoch': epoch,
    #         'state_dict': toy_model.state_dict()},
    #         'train.ckpt')