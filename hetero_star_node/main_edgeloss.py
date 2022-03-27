from model import SuperNet
from numpy.core.numeric import outer
from scipy.sparse import data
import torch
import tqdm
from dataset import WebQnaDataset
from torch_geometric.loader import DataLoader
import numpy as np
from model import ToyNet
from sklearn.metrics import f1_score
#from torch_geometric.nn import to_hetero
import torch
from torch.nn import ReLU
import torch.nn.functional as F
from tqdm import tqdm

#from focalloss import FocalLoss

from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero
from torch.optim.lr_scheduler import _LRScheduler
def evaluate(eval_loader, toy_model):
    toy_model.eval()
    with tqdm(eval_loader, unit="batch") as tepoch:
        val_pred = 0.
        total_vals = 0.
        all_gt = []
        all_pred = []
        loss= 0
        for datum_val in tepoch:                
            #total_vals += datum_val.x.shape[0]
            total_vals += datum_val.x_dict['img_src'].shape[0]
            total_vals += datum_val.x_dict['txt_src'].shape[0]
            datum_val = datum_val.to(device)
            #preds_rand,att=  toy_model.GAT(datum_val.x_dict, datum_val.edge_index_dict)
            pred,att = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
            loss_img_val = criterion(pred['img_src'], datum_val.y_dict['img_src'])
            loss_txt_val = criterion(pred['txt_src'], datum_val.y_dict['txt_src'])
            loss_batch = loss_img_val + loss_txt_val
            loss +=loss_batch.item()
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
            
            all_gt.extend(gt_img.detach().cpu())
            all_gt.extend(gt_txt.detach().cpu())
            all_pred.extend(pred_img.detach().cpu())
            all_pred.extend(pred_txt.detach().cpu())
        all_gt= np.array(all_gt)
        all_pred= np.array(all_pred)
        non_mask_index=  all_gt>=0
        all_gt = all_gt[non_mask_index]
        all_pred = all_pred[non_mask_index]
        f1s = f1_score(all_gt,all_pred)
        val_acc= (val_pred/total_vals)*100
        print("Val loss {} Val Acc :: {}  F1 Score :: {}".format(loss,val_acc, f1s))
        return loss,val_acc, f1s

class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


data_root = "/home/ubuntu/webqa_data"
model_resume_path=""
model_save_path="/home/ubuntu/webqa_data/saved_models_graph/sageconv/model_edgeloss.ckpt"
batch_size = 64
epochs = 500
device = "cuda"

print_step = 270
val_step = 500

webqa_dataset_train = WebQnaDataset(data_root)
webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_train = DataLoader(webqa_dataset_train, batch_size, shuffle=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, batch_size, shuffle=True)

# toy_model = model
#graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src')])

graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','rlink1','ques'), ('img_src','rlink2','ques'), ('txt_src','ti_link','img_src'),
('txt_src','tt_link','txt_src'),('img_src','ii_link','img_src'),'img_src','it_link','txt_src'])

# graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
# ('txt_src','rlink1','ques'), ('img_src','rlink2','ques')])

#loss computed only for these edge types
edge_types_loss= [('ques','link1','txt_src'),('ques','link2','img_src'),('txt_src','ti_link','img_src'),
('txt_src','tt_link','txt_src'),('img_src','ii_link','img_src')]
#edge_types_loss=[]
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
toy_model = SuperNet(edge_types_loss)
toy_model.GAT = to_hetero(toy_model.GAT, graph_meta)
toy_model = toy_model.to(device)

# if(model_resume_path !=""): 
#     toy_model.load_state_dict(torch.load(model_resume_path))

# criterion = torch.nn.BCELoss()
class_weights = torch.tensor([1,15], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(class_weights,ignore_index=-1)
#criterion = FocalLoss(gamma=1, alpha=0.9, reduction='mean')

optimizer = torch.optim.AdamW(toy_model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
#scheduler = NoamLR(optimizer, warmup_steps=10)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
toy_model.train()

g_ctr = 0
train_loss_total=0
for epoch in range(epochs):
    with tqdm(webqa_dataloader_train, unit="batch") as tepoch:
        if(model_save_path !=""):
            torch.save(toy_model,model_save_path)
        if(model_resume_path !=""):
            toy_model= torch.load(model_resume_path)
            toy_model = toy_model.to(device)
        g_ctr = 0
        #val_acc, f1s= evaluate(webqa_dataloader_val,toy_model)
        #print("Epoch :: {} Step :: {} Val Acc :: {}  F1 Score :: ".format(epoch, g_ctr,val_acc, f1s))
        for datum in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            #for idx, datum in enumerate(webqa_dataloader_train):
            g_ctr += 1
            toy_model.train()
            optimizer.zero_grad()
            
            datum = datum.to(device)
            outp,pred_edges = toy_model(datum.x_dict, datum.edge_index_dict)
            
            loss_img = criterion(outp['img_src'], datum.y_dict['img_src'])
            loss_txt = criterion(outp['txt_src'], datum.y_dict['txt_src'])
            loss = loss_img + loss_txt
            edge_loss=0
            for edge_type in edge_types_loss: 
                loss_type= criterion(pred_edges[edge_type],datum[edge_type].edge_label)
                edge_loss +=loss_type.item()
            loss += edge_loss
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), loss_edge= edge_loss , loss_img = loss_img.item(), loss_txt= loss_txt.item())
            # if g_ctr % print_step == 0:
            #     #print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), scheduler.get_last_lr()[0]))
            #     print("Epoch :: {} Step :: {} Loss :: {}  LR :: {:.2e}".format(epoch, g_ctr, loss.item(), optimizer.param_groups[0]['lr']))
            
            if g_ctr % val_step == 0:
                val_loss , val_acc, f1s= evaluate(webqa_dataloader_val,toy_model)
                print("Epoch :: {} Step :: {} Val loss :: {} Val Acc :: {}  F1 Score :: {}".format(epoch, g_ctr,val_loss, val_acc, f1s))
        #scheduler.step(loss_val)
        scheduler.step()
