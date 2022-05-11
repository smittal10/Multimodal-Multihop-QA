import os.path as osp
import typing
from numpy.lib.utils import source
import cv2, base64
import torch
# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData
from PIL import Image
from tqdm import tqdm
import numpy as np
from utils import get_ids, load_bert_feats, load_data, load_clip_feats, load_raw_images
import pandas as pd
import clip

class WebQnaDataset(Dataset):
    def __init__(self, root, val=False, transform=None, pre_transform=None):
        self._WebQna_dataset = load_data()
        self._question_ids = get_ids(self._WebQna_dataset, val=val)
        self._processed_file_names = ["node_"+str(q)+".pt" for q in self._question_ids]
        
        self._bert_feats = load_bert_feats()
        # self._image_feats = load_image_feats()
        # self._image_feats = load_clip_feats()
        self._image_paths, self._lineidx, self._img_tsv_pth = load_raw_images(self._WebQna_dataset)
        _ , self._clip_preprocess = clip.load("ViT-B/32", device="cuda")
        self._caption_feats = {}
        self._question_feats = {}
        # self._img_tsv = open(self._img_tsv_pth,"r")
        #self._question_ids = self._question_ids#[:18000]
        print(len(self._question_ids))
        for id in self._question_ids:
            for k in self._bert_feats[id].keys():
                if k=='Q':
                    self._question_feats[id] = torch.tensor(self._bert_feats[id]['Q'])
                elif 'img' in k:
                    for img_k in self._bert_feats[id][k].keys():
                        self._caption_feats[img_k] = torch.tensor(self._bert_feats[id][k][img_k])
                elif 'txt' in k:
                    for txt_k in self._bert_feats[id][k].keys():
                        self._caption_feats[txt_k] = torch.tensor(self._bert_feats[id][k][txt_k])
        if val == True:
            df = pd.DataFrame()
            df["qids"] = self._question_ids
            df.to_csv("Qids.csv",index=False)
        # print(self._question_feats[t_id])      
        # print(self._question_feats[t2_id])      
        # print(self._WebQna_dataset[t_id]['Q'])
        # print(t_id)
        # for id in self._question_feats:
        #     if len(self._WebQna_dataset[id]['img_posFacts']) > 1:
        #         print(self._WebQna_dataset[id]['img_posFacts'])
        #         break
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return self._processed_file_names
        # return []
    def len(self):
        return len(self._question_ids)
        pass
    
    def get_edges(self, num_srcs, num_targets):
        source_nodes = []
        for i in range(num_srcs):
            source_nodes += [i]*(num_targets)        
        target_nodes = [i for i in range(num_targets)]*num_srcs

        edge_index = torch.tensor([source_nodes, target_nodes], 
                    dtype=torch.long)
        return edge_index

    def get(self, idx:int)-> Data:
        y = [] # TODO: Remove
        y_img = []
        y_txt = []
        x = [] # TODO: Remove
        x_img = []
        x_txt = []
        x_img_cap = []
        id = self._question_ids[idx]
        ques_feat = self._question_feats[id]
        x.append(ques_feat.unsqueeze(0))
        for pos_image_dict in self._WebQna_dataset[id]['img_posFacts']:
            image_id = pos_image_dict['image_id']

            ##############################################
            # #raw image feat- input to clip
            # pos_image_feat = self._image_feats[image_id]
            with open(self._img_tsv_pth, "r") as fp:
                fp.seek(self._lineidx[int(image_id)%10000000])
                imgid, img_base64 = fp.readline().strip().split('\t')
                image = cv2.imdecode(np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    image = np.zeros((512,512,3), dtype=np.uint8).float()
                image = image[:,:,::-1]
                pos_image_feat = self._clip_preprocess(Image.fromarray(image)).unsqueeze(0)
            ##############################################
            pos_image_caption_feat = self._caption_feats[image_id].unsqueeze(0)
            # node_features = torch.cat((pos_image_caption_feat, pos_image_feat),dim=-1).unsqueeze(0) # do in forward call
            #x.append(node_features) # TODO: Remove
            x_img.append(pos_image_feat)
            x_img_cap.append(pos_image_caption_feat)
            y.append(1) # TODO: Remove
            y_img.append(1)
            # y.append([0,1])
        for neg_image_dict in self._WebQna_dataset[id]['img_negFacts']:
            image_id = neg_image_dict['image_id']
            ##############################################
            # #raw image feat- input to clip
            # neg_image_feat = self._image_feats[image_id]
            with open(self._img_tsv_pth, "r") as fp:
                fp.seek(self._lineidx[int(image_id)%10000000])
                imgid, img_base64 = fp.readline().strip().split('\t')
                image = cv2.imdecode(np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    image = np.zeros((512,512,3), dtype=np.uint8)
                image = image[:,:,::-1]
                neg_image_feat = self._clip_preprocess(Image.fromarray(image)).unsqueeze(0)
            ##############################################
            neg_image_caption_feat = self._caption_feats[image_id].unsqueeze(0)

            # node_features = torch.cat((neg_image_caption_feat, neg_image_feat),dim=-1).unsqueeze(0)

            #print(f"Img facts: {node_features.shape}")
            x_img.append(neg_image_feat)
            x_img_cap.append(neg_image_caption_feat)
            y.append(0) # TODO: Remove
            y_img.append(0)
        for neg_txt_dict in self._WebQna_dataset[id]['txt_negFacts']:
            snippet_id = neg_txt_dict['snippet_id']
            neg_txt_feat = self._caption_feats[snippet_id]
            node_features = neg_txt_feat.unsqueeze(0)
            #print(f"X facts: {node_features.shape}")
            x_txt.append(node_features)
            y_txt.append(0)
        for pos_txt_dict in self._WebQna_dataset[id]['txt_posFacts']:
            snippet_id = pos_txt_dict['snippet_id']
            pos_txt_feat = self._caption_feats[snippet_id]
            node_features = pos_txt_feat.unsqueeze(0)
            x_txt.append(node_features)
            y_txt.append(1)
        #print(node_features.shape)
        if len(y_txt) == 0:
            y_txt = [0]
            txt_feat = torch.zeros(768)
            x_txt = [txt_feat.unsqueeze(0)]
        
        if len(y_img) == 0:
            y_img = [0]
            # img_feat = torch.zeros(2048+768) # resnet
            cap_feat = torch.zeros(768) #clip
            x_img = [torch.zeros((3, 224, 224), dtype=torch.float16).unsqueeze(0)]
            x_img_cap = [cap_feat.unsqueeze(0)]

            # y.append([1,0])
        # for sh in x:
        #     # print(sh[0].shape[0])
        #     if sh[0].shape[0] != 768:
        #         print("missing q")
        #         break
        #     if sh[2].shape[0] != 768:
        #         print("missing cap")
        #         break
        
        # node_idx = [i for i in range(len(y))]
        # # node_idx_idx = np.arange(start=0,stop=len(y),step=1)
        # # node_idx = list(np.random.permutation(node_idx))
        
        # # print(node_idx)
        # source_nodes = []
        # for i in range(len(y)):
        #     source_nodes += [i]*(len(y)-1)
        # target_nodes = []
        # for i in range(len(y)):
        #     target_nodes += node_idx[:i] + node_idx[i+1:]
        
        # # source_nodes = node_idx[:-1]
        # # target_nodes = node_idx[1:]
        # # print(len(source_nodes), len(target_nodes))
        # # assert False
        # # edge_index = torch.tensor([source_nodes + target_nodes, target_nodes + source_nodes], dtype=torch.long)
        # edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        # x = torch.cat(x,dim=0)
        # # y = torch.FloatTensor(y)
        # y = torch.LongTensor(y)
        # # y = torch.IntTensor(y)
        # #data = Data(x=x, edge_index=edge_index, y=y)
        num_ques = len(x)
        num_txt_srcs = len(x_txt)
        num_img_srcs = len(x_img)
        edge_index_qt = self.get_edges(num_ques, num_txt_srcs)
        edge_index_qi = self.get_edges(num_ques, num_img_srcs)
        edge_index_tq = self.get_edges(num_txt_srcs, num_ques)
        edge_index_iq = self.get_edges(num_img_srcs, num_ques)

        data = HeteroData()
        data['ques'].x = torch.cat(x, dim=0)
        data['img_src'].x = torch.cat(x_img, dim=0)
        data['img_src_cap'].x = torch.cat(x_img_cap,dim=0)
        data['txt_src'].x = torch.cat(x_txt, dim=0)
        data['img_src'].y = torch.LongTensor(y_img)
        data['txt_src'].y = torch.LongTensor(y_txt)
        data['ques','link1','txt_src'].edge_index = edge_index_qt
        data['ques','link2','img_src'].edge_index = edge_index_qi
        data['txt_src','link3','ques'].edge_index = edge_index_tq
        data['img_src','link4','ques'].edge_index = edge_index_iq

        return data

    # def process(self):

    #     for id in tqdm(self._question_ids):
    #         # y = []
    #         # x = []
    #         # ques_feat = torch.tensor(self._question_feats[id])
    #         # for pos_image_dict in self._WebQna_dataset[id]['img_posFacts']:
    #         #     image_id = pos_image_dict['image_id']
    #         #     pos_image_feat = self._image_feats[image_id]
    #         #     pos_image_caption_feat = self._caption_feats[image_id]
    #         #     node_features = [ques_feat,pos_image_feat, pos_image_caption_feat]
    #         #     x.append(node_features)
    #         #     y.append(1)
    #         # for pos_image_dict in self._WebQna_dataset[id]['img_negFacts']:
    #         #     image_id = pos_image_dict['image_id']
    #         #     pos_image_feat = self._image_feats[image_id]
    #         #     pos_image_caption_feat = self._caption_feats[image_id]
    #         #     node_features = [ques_feat,pos_image_feat, pos_image_caption_feat]
    #         #     x.append(node_features)
    #         #     y.append(0)
    #         # for sh in x:
    #         #     # print(sh[0].shape[0])
    #         #     if sh[0].shape[0] != 768:
    #         #         print("missing q")
    #         #         break
    #         #     if sh[2].shape[0] != 768:
    #         #         print("missing cap")
    #         #         break
    #         # node_idx = [i for i in range(len(y))]
    #         # source_nodes = node_idx[:-1]
    #         # target_nodes = node_idx[1:]
    #         # edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            
    #         # y = torch.FloatTensor(y)
    #         # data = Data(x=x, edge_index=edge_index, y=y)
            
    #         # torch.save(data, osp.join(self.processed_dir, f'node_{id}.pt'))
