import os.path as osp
import typing
from numpy.lib.utils import source

import torch
# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData

from tqdm import tqdm
import numpy as np
from utils import get_ids, load_bert_feats, load_data, load_image_feats

import torch_geometric.transforms as T


class WebQnaDataset(Dataset):
    def __init__(self, root, val=False, transform=None, pre_transform=None):
        self._WebQna_dataset = load_data()
        self._question_ids = get_ids(self._WebQna_dataset, val=val)
        self._processed_file_names = ["node_"+str(q)+".pt" for q in self._question_ids]
        
        self._bert_feats = load_bert_feats()
        self._image_feats = load_image_feats()
                
        self._caption_feats = {}
        self._question_feats = {}
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

    def mask_negatives(self,y,mask_prob=0.7):
        #y= y.squeeze(0)
        y_neg = (y== 0).nonzero().squeeze(1)
        num_samples_msk= int(mask_prob*y_neg.shape[-1])
        y_neg_mskindex= torch.randperm(y_neg.shape[-1])[:num_samples_msk]
        #y_txt_neg_masked= torch.where(y_txt_neg_mskindex,y_txt_neg,torch.tensor(-1.0))
        y[y_neg_mskindex]= -1
        return y


    def get(self, idx:int)-> Data:
        y = [] # TODO: Remove
        y_img = []
        y_txt = []
        x = [] # TODO: Remove
        x_img = []
        x_txt = []
        id = self._question_ids[idx]
        ques_feat = self._question_feats[id]
        x.append(ques_feat.unsqueeze(0))
        for pos_image_dict in self._WebQna_dataset[id]['img_posFacts']:
            image_id = pos_image_dict['image_id']
            pos_image_feat = self._image_feats[image_id]
            pos_image_caption_feat = self._caption_feats[image_id]
            # node_features = torch.cat((ques_feat, pos_image_feat, pos_image_caption_feat),dim=-1).unsqueeze(0)
            node_features = torch.cat((pos_image_caption_feat, pos_image_feat),dim=-1).unsqueeze(0)
            #x.append(node_features) # TODO: Remove
            x_img.append(node_features)
            y.append(1) # TODO: Remove
            y_img.append(1)
            # y.append([0,1])
        for neg_image_dict in self._WebQna_dataset[id]['img_negFacts']:
            image_id = neg_image_dict['image_id']
            neg_image_feat = self._image_feats[image_id]
            neg_image_caption_feat = self._caption_feats[image_id]
            # node_features = torch.cat((ques_feat, neg_image_feat, neg_image_caption_feat),dim=-1).unsqueeze(0)
            node_features = torch.cat((neg_image_caption_feat, neg_image_feat),dim=-1).unsqueeze(0)
            #x.append(node_features) # TODO: Remove
            #print(f"Img facts: {node_features.shape}")
            x_img.append(node_features)
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
            img_feat = torch.zeros(2048+768)
            x_img = [img_feat.unsqueeze(0)]
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
        #src-src
        edge_index_ti = self.get_edges(num_txt_srcs, num_img_srcs)
        edge_index_it = self.get_edges(num_img_srcs,num_txt_srcs)
        edge_index_tt = self.get_edges(num_txt_srcs, num_txt_srcs)
        edge_index_ii = self.get_edges(num_img_srcs, num_img_srcs)

        data = HeteroData()
        data['ques'].x = torch.cat(x, dim=0)
        data['img_src'].x = torch.cat(x_img, dim=0)
        data['txt_src'].x = torch.cat(x_txt, dim=0)
        
        data['ques','link1','txt_src'].edge_index = edge_index_qt
        data['ques','link2','img_src'].edge_index = edge_index_qi
        data['txt_src','rlink1','ques'].edge_index = edge_index_tq
        data['img_src','rlink2','ques'].edge_index = edge_index_iq
        ####src-src
        data['txt_src','ti_link','img_src'].edge_index = edge_index_ti
        data['txt_src','tt_link','txt_src'].edge_index = edge_index_tt
        data['img_src','ii_link','img_src'].edge_index = edge_index_ii
        data['img_src','it_link','txt_src'].edge_index = edge_index_it
        
        #####
        y_txt = torch.LongTensor(y_txt)
        y_img = torch.LongTensor(y_img)


        # #mask negatives with -1 for sampling
        y_txt= self.mask_negatives(y_txt,mask_prob=0.85)
        y_img= self.mask_negatives(y_img,mask_prob=0.85)
        
        data['img_src'].y = torch.LongTensor(y_img)
        data['txt_src'].y = torch.LongTensor(y_txt)

        data['ques'].num_img_srcs =  num_img_srcs
        data['ques'].num_txt_srcs = num_txt_srcs

        data['ques','link1','txt_src'].edge_label = y_txt
        data['ques','link2','img_src'].edge_label = y_img
        data['txt_src','rlink','ques'].edge_label = y_txt
        data['img_src','rlink2','ques'].edge_label = y_img

        
        #src-src edge labels
        y_txt = torch.unsqueeze(y_txt,0)
        y_img = torch.unsqueeze(y_img,0)

        

        data['txt_src','ti_link','img_src'].edge_label = torch.mm(y_txt.T,y_img).flatten()
        data['txt_src','tt_link','txt_src'].edge_label = torch.mm(y_txt.T,y_txt).flatten()
        data['img_src','ii_link','img_src'].edge_label = torch.mm(y_img.T,y_img).flatten()
        data['img_src','it_link','txt_src'].edge_label = torch.mm(y_img.T,y_txt).flatten()


        # data = T.ToUndirected()(data)
        # data = T.AddSelfLoops()(data)
        # data = T.NormalizeFeatures()(data)

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
