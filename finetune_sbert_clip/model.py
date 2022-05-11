import torch
import torch.nn as nn
from sentence_transformers import util

class combinedModel(nn.Module):
    def __init__(self,sbert,clip_model):
        super().__init__()
        self.sbert = sbert
        self.clip = clip_model
        self.mlp_img = nn.Sequential(nn.Linear(2304,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,2))
        self.mlp_txt = nn.Sequential(nn.Linear(1536,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,2))
    
    def forward_img(self, batch, device):
        ques = self.sbert.tokenize(batch[0])
        ques = util.batch_to_device(ques,device)
        ques  = self.sbert(ques)['sentence_embedding']
        caption = self.sbert.tokenize(batch[1])
        caption = util.batch_to_device(caption,device)
        caption = self.sbert(caption)['sentence_embedding']
        image = self.clip(batch[2].to(device))
        out = self.mlp_img(torch.cat((ques,caption,image),dim=-1))
        return out
    def forward_txt(self, batch, device):
        ques = self.sbert.tokenize(batch[0])
        ques = util.batch_to_device(ques,device)
        ques  = self.sbert(ques)['sentence_embedding']
        passage = self.sbert.tokenize(batch[1])
        passage = util.batch_to_device(passage,device)
        passage = self.sbert(passage)['sentence_embedding']
        out = self.mlp_txt(torch.cat((ques,passage),dim=-1))
        return out

