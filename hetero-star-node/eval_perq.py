from re import T
from numpy.core.numeric import outer
from scipy.sparse import data
import torch
import pandas as pd
from dataset import WebQnaDataset
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import tqdm
from tqdm import tqdm
import numpy as np
from model import ToyNet
from sklearn.metrics import f1_score
from focalloss import FocalLoss
from torch_geometric.nn import to_hetero
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
val_step = 2000


# key = torch.tensor([10])
# d = webqa_dataset.get(idx=10)

toy_model = ToyNet()


# graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
# ('txt_src','rlink1','ques'), ('img_src','rlink2','ques'), ('txt_src','ti_link','img_src'),
# ('txt_src','tt_link','txt_src'),('img_src','ii_link','img_src'),('img_src','it_link','txt_src')])


graph_meta = (['txt_src', 'img_src', 'ques'], [('ques','link1','txt_src'), ('ques','link2','img_src'), 
('txt_src','link3','ques'), ('img_src','link4','ques')])

# graph_meta = (['txt_src', 'img_src', 'ques', 'ent'], [('ques','link1','txt_src'), ('ques','link2','img_src'), ('ques','link3','ent'),
# ('txt_src','rlink1','ques'), ('img_src','rlink2','ques'), ('ent','rlink3','ques'), ('txt_src','ti_link','img_src'),
# ('txt_src','tt_link','txt_src'),('img_src','ii_link','img_src'),('img_src','it_link','txt_src'), 
# ('txt_src','link1','ent'),('img_src','link2','ent'),('ent','rlink1','txt_src'), ('ent','rlink2','img_src'), 
# ('ent','link1','ent')
# ])

#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src')])
#graph_meta = (['txt_src', 'img_src'], [('txt_src','link1','txt_src'), ('img_src','link2','img_src'), 
#('txt_src','link3','img_src'), ('img_src','link4','txt_src')])
def evaluate_perquestion(eval_loader, toy_model): 
    toy_model.eval()
    precision_overall =0
    fscore_txt_overall =0
    fscore_img_overall =0
    fscore_overall =0

    total_ques_count =0

    result_dict={"qid":[],"cat":[],"prec":[],"rec":[],"f1":[],"f1_txt":[],"f1_img":[]}


    with tqdm(eval_loader, unit="batch") as tepoch:
        with torch.no_grad():
            for datum_val in tepoch: 
                datum_val = datum_val.to(device)
                pred = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
                
                pred_img = torch.argmax(pred['img_src'], dim=-1).detach().cpu().numpy()
                pred_txt = torch.argmax(pred['txt_src'], dim=-1).detach().cpu().numpy()

                num_srcs_txt = datum_val.num_txt_src_dict["ques"].detach().cpu().numpy()
                num_srcs_imgs = datum_val.num_img_src_dict["ques"].detach().cpu().numpy()

                gt_img = datum_val.y_dict['img_src'].detach().cpu().numpy()
                gt_txt = datum_val.y_dict['txt_src'].detach().cpu().numpy()

                ques_ids = datum_val.id_dict["ques"]
                ques_cat = datum_val.category_dict["ques"]
                num_ques_batch= len(ques_ids)
                total_ques_count += num_ques_batch
                curimg_idx=0
                curtxt_idx=0
                for idx, qid in enumerate(ques_ids):
                    
                    pred_imgs_qid= pred_img[curimg_idx: curimg_idx+num_srcs_imgs[idx]]
                    pred_txt_qid= pred_txt[curtxt_idx: curtxt_idx+num_srcs_txt[idx]]


                    gt_imgs_qid= gt_img[curimg_idx: curimg_idx+num_srcs_imgs[idx]]
                    gt_txt_qid= gt_txt[curtxt_idx: curtxt_idx+num_srcs_txt[idx]]

                    curimg_idx = curimg_idx+num_srcs_imgs[idx]
                    curtxt_idx = curtxt_idx+num_srcs_txt[idx]

                    gt= np.concatenate((gt_imgs_qid, gt_txt_qid))
                    pred= np.concatenate((pred_imgs_qid,pred_txt_qid))


                    precision = precision_score(gt, pred)
                    recall = recall_score(gt, pred)
                    fscore_txt= f1_score(gt_txt_qid,pred_txt_qid)
                    f1score_img = f1_score(gt_imgs_qid,pred_imgs_qid)
                    fscore = f1_score(gt, pred)
                    
                   
                    result_dict["qid"].append(qid)
                    #result_dict["qid"].append(qid)
                    result_dict["cat"].append(ques_cat[idx])
                    result_dict["prec"].append(precision)
                    result_dict["rec"].append(recall)
                    result_dict["f1_img"].append(f1score_img)
                    result_dict["f1_txt"].append(fscore_txt)
                    result_dict["f1"].append(fscore)                    
            
            df = pd.DataFrame.from_dict(result_dict)
            df.to_csv("hetero-star-node/ques_f1_contrast.csv")
            print("f1:{} prec:{},recall:{}".format(df["f1"].mean(), df["prec"].mean(),df["rec"].mean()))
            print("f1:{},  f1_txt:{}, f1_img:{}".format(df["f1"].mean(), df["f1_txt"].mean(),df["f1_img"].mean()))
            return df["f1"].mean(), df["f1_txt"].mean(), df["f1_img"].mean()


def evaluate(eval_loader, toy_model):
    toy_model.eval()
    
    save_gt_ =[]
    save_pred_ =[]
    micro_f1_sum=0
    with tqdm(eval_loader, unit="batch") as tepoch:
        val_pred = 0.
        total_vals = 0.
        all_gt = []
        all_pred = []
        img_pred=[]
        txt_pred=[]
        img_gt=[]
        txt_gt=[]
        with torch.no_grad():
            for datum_val in tepoch:  
                #gc.collect()              
                #total_vals += datum_val.x.shape[0]
                total_vals += datum_val.x_dict['img_src'].shape[0]
                total_vals += datum_val.x_dict['txt_src'].shape[0]
                datum_val = datum_val.to(device)
                #preds_rand=  toy_model(datum_val.x_dict, datum_val.edge_index_dict)
                pred = toy_model(datum_val.x_dict, datum_val.edge_index_dict)
                
                pred_img = torch.argmax(pred['img_src'], dim=-1)
                pred_txt = torch.argmax(pred['txt_src'], dim=-1)
                # gt = torch.argmax(datum_val.y, dim=-1)
                #gt = datum_val.y
                gt_img = datum_val.y_dict['img_src']
                gt_txt = datum_val.y_dict['txt_src']
                #total_items_ques= datum["num_txt_srcs"] + datum["num_img_srcs"]
                # print(datum_val.x.shape[0],torch.sum(outp==0))
                # val_pred += torch.sum(outp==datum_val.y).detach().cpu().item()
                #val_pred += torch.sum(pred==gt).detach().cpu().item()
                val_pred_img = torch.sum(pred_img==gt_img).detach().cpu().item()
                val_pred_txt = torch.sum(pred_txt==gt_txt).detach().cpu().item()
                val_pred += val_pred_img + val_pred_txt
                gt_img= (gt_img.detach().cpu().numpy())
                gt_txt= gt_txt.detach().cpu().numpy()
                pred_img = pred_img.detach().cpu().numpy()
                pred_txt= pred_txt.detach().cpu().numpy()
                micro_f1_sum += f1_score(np.append(gt_img,gt_txt),np.append(pred_img, pred_txt) )

                save_gt_.extend(list(gt_img))
                save_gt_.extend(list(gt_txt))
                save_pred_.extend(list(pred_img))
                save_pred_.extend(list(pred_txt))

                df = pd.DataFrame()
                df["Pred"] = save_pred_
                df["GT"] = save_gt_
                df.to_csv("Predictions.csv", index=False)
                #all_gt.extend(datum_val.y.detach().cpu())
                #all_gt.extend(gt.detach().cpu())
                #all_pred.extend(pred.detach().cpu())
                img_gt.extend(gt_img)
                img_pred.extend(pred_img)

                txt_pred.extend(pred_txt)
                txt_gt.extend(gt_txt)

            
                all_gt.extend(gt_img)
                all_gt.extend(gt_txt)
                all_pred.extend(pred_img)
                all_pred.extend(pred_txt)
        all_gt= np.array(all_gt)
        all_pred= np.array(all_pred)
        non_mask_index=  all_gt>=0
        all_gt = all_gt[non_mask_index]
        all_pred = all_pred[non_mask_index]
        f1s_img = f1_score(img_gt,img_pred)
        f1s_text = f1_score(txt_gt,txt_pred)

        f1s = [f1_score(all_gt,all_pred),f1s_img, f1s_text]
        val_acc= accuracy_score(all_gt,all_pred)
        print("microf1:: {}".format(micro_f1_sum/4966))
        print("Val Acc :: {}  F1 Score :: {}".format(val_acc, f1s))
        return val_acc, f1s

toy_model = to_hetero(toy_model, graph_meta,aggr="sum")
toy_model = torch.load("hetero-star-node/best_node_finetuned.ckpt",map_location="cpu")
toy_model = toy_model.to(device)

webqa_dataset_val = WebQnaDataset(data_root, val=True)
webqa_dataloader_val = DataLoader(webqa_dataset_val, 64, shuffle=False)


toy_model = toy_model.to(device)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, threshold=0.01, min_lr=1e-6)
toy_model.train()
 
g_ctr = 0
best_f1s = 0

best_f1=0
#val_acc, f1s= evaluate(webqa_dataloader_val,toy_model)
#print("Val Acc :: {} LR :: {}  F1 Score :: {} f1_img :: {} f1_txt :: {} ".format(,val_acc,optimizer.param_groups[0]['lr'],  f1s[0], f1s[1],f1s[2]))


# val_acc, f1s= evaluate(webqa_dataloader_val,toy_model)
# print("Val Acc :: {} LR :: {}  F1 Score :: {} f1_img :: {} f1_txt :: {} ".format(val_acc,optimizer.param_groups[0]['lr'],  f1s[0], f1s[1],f1s[2]))
evaluate_perquestion(webqa_dataloader_val,toy_model)

