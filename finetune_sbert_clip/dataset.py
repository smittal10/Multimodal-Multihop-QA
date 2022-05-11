import json, base64
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class QSrcDataset(Dataset):
    def __init__(self, json_path, img_tsv_path, lineidx, preprocess):
        super().__init__()
        with open(json_path,'r') as fp:
            self.dataset = json.load(fp)
            self._img_tsv_pth = img_tsv_path
            self._lineidx = lineidx
            self._clip_preprocess = preprocess

    def __getitem__(self, index):
        ques_dict = self.dataset[index]
        
        if 'image_id' in ques_dict:
            image_id = ques_dict['image_id']
            with open(self._img_tsv_pth, "r") as fp:
                fp.seek(self._lineidx[int(image_id)%10000000])
                imgid, img_base64 = fp.readline().strip().split('\t')
                image = cv2.imdecode(np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    image = np.zeros((512,512,3), dtype=np.uint8)
                image = image[:,:,::-1]
                image = self._clip_preprocess(Image.fromarray(image)).half()
            question = ques_dict[' Q']
            caption = ques_dict['caption']
            label = torch.LongTensor([int(ques_dict['label'])])
            return {'img_src': (question, caption, image,label)}
        else:
            question = ques_dict[' Q']
            passage = ques_dict['fact']
            label = torch.LongTensor([int(ques_dict['label'])])
            return {'txt_src': (question, passage,label)}
    
    def __len__(self):
        return len(self.dataset)
            
def my_collate_fn(data):
    to_return = {}

    for d in data:
        k = list(d.keys())[0]
        if k in to_return:
            to_return[k].append(d[k])
        else:
            to_return[k] = [d[k]]
    
    for k, v in to_return.items():
        if k == 'img_src' and v!= []:
            question, caption, image,label = zip(*v)
            to_return[k] = list(question), list(caption), torch.stack(image), torch.stack(label)
        elif k == 'txt_src' and v!= []:
            question, passage,label = zip(*v)
            to_return[k] = list(question), list(passage), torch.stack(label)
    
    return to_return

    
        


    # _, labels, lengths = zip(*data)
    # max_len = max(lengths)
    # n_ftrs = data[0][0].size(1)
    # features = torch.zeros((len(data), max_len, n_ftrs))
    # labels = torch.tensor(labels)
    # lengths = torch.tensor(lengths)

    # for i in range(len(data)):
    #     j, k = data[i][0].size(0), data[i][0].size(1)
    #     features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    # return features.float(), labels.long(), lengths.long()


