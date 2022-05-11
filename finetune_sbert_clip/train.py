import torch
from dataset import QSrcDataset, my_collate_fn
import numpy as np
from model import combinedModel
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from sentence_transformers import SentenceTransformer
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def convert_models_to_fp32(model): 
    for p in model.parameters():
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


device = "cuda" if torch.cuda.is_available() else "cpu"

#load the clip model
clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
clip_model = clip_model.visual
clip.model.convert_weights(clip_model)
clip_model.to(device)

#load the sbert model
sbert = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
print("Max Sequence Length:", sbert.max_seq_length)
#Change the length
sbert.to(device)
##load data
batch_size = 24
epochs = 500
print_step = 500
val_step = 15000

img_tsv = "/home/ubuntu/data/imgs.tsv"
lineidx_path = '/home/ubuntu/data/imgs.lineidx'
with open(lineidx_path, "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

webqa_dataset_train = QSrcDataset("/home/ubuntu/data/pairwise/train.json",img_tsv,lineidx,preprocess)
webqa_dataset_val = QSrcDataset("/home/ubuntu/data/pairwise/val.json",img_tsv,lineidx,preprocess)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size=batch_size,collate_fn=my_collate_fn, shuffle=True,num_workers=6)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size=batch_size,collate_fn=my_collate_fn, shuffle=False,num_workers=6)

model = combinedModel(sbert,clip_model)

model.to(device)

class_weights = torch.tensor([1,10], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights)
optim_list = [{'params': clip_model.parameters(),'lr': 1e-7},{'params': sbert.parameters(),'lr': 1e-5},{'params': model.mlp_img.parameters()},{'params': model.mlp_txt.parameters()}]
optimizer = torch.optim.AdamW(optim_list, lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
model.train()
 
g_ctr = 0
best_f1s = 0
local_batch_size = batch_size
total_batch_size = 144
grad_acc_steps = total_batch_size / local_batch_size
for epoch in range(epochs):
    g_ctr = 0
    for idx, datum in enumerate(webqa_dataloader_train):
        g_ctr += 1
        # clip_model.train()
        model.train()
        # optimizer.zero_grad()
        fg_i, fg_t = 0, 0
        if 'img_src' in datum.keys():
            outp_img = model.forward_img(datum['img_src'],device)
            loss_img = criterion(outp_img, datum['img_src'][-1].view(-1).to(device))
            loss_img = loss_img/grad_acc_steps
            fg_i = 1
            # loss_img.backward()
        if 'txt_src' in datum.keys():
            outp_txt = model.forward_txt(datum['txt_src'],device)
            loss_txt = criterion(outp_txt, datum['txt_src'][-1].view(-1).to(device))
            loss_txt = loss_txt/grad_acc_steps
            fg_t = 1
            # loss_txt.backward()
        if fg_i==1 and fg_t==1:
            loss = loss_img + loss_txt
        elif fg_i == 0:
            loss = loss_txt
        else:
            loss = loss_img
        loss.backward()
        if (idx+1)%grad_acc_steps == 0:
            convert_models_to_fp32(clip_model)
            optimizer.step()
            optimizer.zero_grad()
            clip.model.convert_weights(clip_model)
        if g_ctr % print_step == 0:
            #print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
            print("Epoch :: {} Step :: {} Loss Image :: {}  Loss Text :: {} LR :: {:.2e}".format(epoch, g_ctr, loss_img.item(), loss_txt.item(), optimizer.param_groups[0]['lr']))
        
    ##validation
        if g_ctr % val_step == 0:
            clip_model.eval()
            sbert.eval()
            model.eval()
            val_pred = 0.
            total_vals = 0.
            val_loss = 0
            all_gt = []
            all_pred = []
            save_gt_ = []
            save_pred_ = []
            for idx, datum_val in enumerate(webqa_dataloader_val):
                total_vals += batch_size
                with torch.no_grad():
                    if 'img_src' in datum_val.keys():
                        outp_img = model.forward_img(datum_val['img_src'],device)
                        val_loss += criterion(outp_img, datum_val['img_src'][-1].view(-1).to(device)).item()
                        pred_img = torch.argmax(outp_img, dim=-1).detach().cpu()
                        gt_img = datum_val['img_src'][-1].view(-1)
                        val_pred += torch.sum(pred_img==gt_img).detach().cpu().item()
                        all_gt.extend(gt_img.detach().cpu())
                        all_pred.extend(pred_img)
                    if 'txt_src' in datum_val.keys():
                        outp_txt = model.forward_txt(datum_val['txt_src'],device)
                        val_loss += criterion(outp_txt, datum_val['txt_src'][-1].view(-1).to(device)).item()
                        pred_txt = torch.argmax(outp_txt, dim=-1).detach().cpu()
                        gt_txt = datum_val['txt_src'][-1].view(-1)
                        val_pred += torch.sum(pred_txt==gt_txt).detach().cpu().item()
                        all_gt.extend(gt_txt.detach().cpu())
                        all_pred.extend(pred_txt)
                
            f1s = f1_score(all_gt,all_pred)
            if f1s > best_f1s:
                # save_preds
                torch.save(clip_model,"fine_tuned_clip.ckpt")
                torch.save(sbert,"finetuned_sbert.ckpt")
                torch.save(model.mlp_img,"mlp_img.ckpt")
                torch.save(model.mlp_txt,"mlp_txt.ckpt")
                best_f1s = f1s
            print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr, (val_pred/total_vals)*100), f1s)
            print("val loss :: {}".format(val_loss/len(webqa_dataloader_val)))
    #scheduler.step(loss_val)
    scheduler.step()
