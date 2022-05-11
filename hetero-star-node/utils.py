import json, time, os, base64
import os.path as osp
import numpy as np
from collections import Counter
import torch

from easydict import EasyDict as edict
import pickle
import gc

cfg = {}
cfg['trainval_path'] = "/home/ubuntu/data/WebQA_train_val.json"
cfg['bert_feat_path'] = "/home/ubuntu/data/sbert2/sbert"
# cfg['bert_img_filename'] = 'ImgQueries_embeddings.pkl'
# cfg['bert_txt_filename'] = 'TxtQueries_embeddings.pkl' /home/ubuntu/data/sbert2/sbert/ImgQueries_embeddings.pt
cfg['bert_img_filename'] = 'ImgQueries_embeddings.pt'
cfg['bert_txt_filename'] = 'TxtQueries_embeddings.pt'

cfg['image_feat_path'] = "/home/ubuntu/data/resnet/save_resnet152_embed_pos_neg/"
cfg['image_saved_ckpt_path'] = "/home/ubuntu/data/resnet/image_feats.ckpt"
cfg['image_clip_feats_path'] = "/home/ubuntu/data/clip/img_feats_finetuned.pt"
# cfg['bert_feat_filename'] = 'ImgQueries_embeddings.pkl'

def load_data(args=None):
    trainval_path =  cfg['trainval_path']
    dataset = json.load(open(trainval_path, "r"))
    
    if args is None or args.verbose:
        print("Data Split: ", Counter([dataset[k]['split'] for k in dataset]))
        print("Total queries: ", len(set([dataset[k]['Guid'] for k in dataset])))
        print("Query Types: ", Counter([dataset[k]['Qcate'] for k in dataset]))
    return dataset


def get_ids(dataset, val=False, args=None,):
    if val:
        ids = [id for id in dataset if dataset[id]['split']=='val']
    else:
        ids = [id for id in dataset if dataset[id]['split']=='train']
    return ids


def load_bert_imgfeats(args=None,):
    bert_files = osp.join(cfg['bert_feat_path'],cfg['bert_img_filename'])
    
    # with open(bert_files, 'rb') as f:
    #     bert_feats = pickle.load(f)
        # print(bert_feats)
    bert_feats = torch.load(bert_files)
    return bert_feats

def load_bert_txtfeats(args=None,):
    bert_files = osp.join(cfg['bert_feat_path'],cfg['bert_txt_filename'])
    
    # with open(bert_files, 'rb') as f:
    #     bert_feats = pickle.load(f)
        # print(bert_feats)
    bert_feats = torch.load(bert_files)
    return bert_feats

def load_bert_feats(args=None,):
    img_dict = load_bert_imgfeats()
    txt_dict = load_bert_txtfeats()
    all_dict = {**img_dict, **txt_dict}
    return all_dict

def load_image_feats(args=None,):
    if osp.isfile(cfg['image_saved_ckpt_path']):
        image_features = torch.load(cfg['image_saved_ckpt_path'])
        
    else:
        file_name_tensor = []
        file_feat_tensor = []
        folder_files = sorted(os.listdir(cfg['image_feat_path']))
        
        for image_f in folder_files:
            if image_f[-3:]=='pkl':
                print(image_f, 'pkl')
                with open(osp.join(cfg['image_feat_path'],image_f), 'rb') as f:
                    image_name = pickle.load(f)
                    image_name = [i.item() for i in image_name]
                    file_name_tensor.extend(image_name)
            elif image_f[-2:]=='pt':
                print(image_f, 'pt')
                with open(osp.join(cfg['image_feat_path'],image_f), 'rb') as f:
                    image_feats = torch.load(f)
                    file_feat_tensor.append(image_feats)
        
        file_feat_tensor = torch.cat(file_feat_tensor,dim=0)
        
        image_features = {file_name_tensor[i] : file_feat_tensor[i,:] for i in range(len(file_name_tensor))}
        # torch.save(image_features,cfg['image_saved_ckpt_path'])
    return image_features

def load_clip_feats():
    image_feats =  torch.load(cfg['image_clip_feats_path'])
    return image_feats

def load_raw_images(dataset):
    image_paths=[]
    for i,k in enumerate(dataset):
        # if dataset[k]['Qcate']!='text': #dataset[k]['split']=='val' and 
        for f in dataset[k]['img_posFacts']:
            image_paths.append(f['image_id'])
        for f in dataset[k]['img_negFacts']:
            image_paths.append(f['image_id'])
    image_paths = list(set(image_paths))
    image_paths.sort()
    print(len(image_paths))

    lineidx_path = '/home/ubuntu/data/imgs.lineidx'
    image_tsv_path = '/home/ubuntu/data/imgs.tsv'
    with open(lineidx_path, "r") as fp_lineidx:
        lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
    return image_paths, lineidx, image_tsv_path
    
