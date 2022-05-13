# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:29:55 2022

@author: remit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from transformers import  BertModel,  BertTokenizerFast
import sys
from torch.utils.data import DataLoader, Dataset
# from cuml.neighbors import NearestNeighbors
class TokenizedDataset(Dataset):
    
    def __init__(self,df,max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        
    def __getitem__(self, index):
        line=self.df.iloc[index]
        
        inputs = self.tokenizer.encode_plus(line.text,
                                            padding="max_length",
                                            max_length=self.max_len,
                                            add_special_tokens=True,
                                            return_token_type_ids=True,
                                            truncation=True)
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])
        return ids,mask
    
    def __len__(self):
        return self.df.shape[0]
    
class Cat2VecModel(nn.Module):
    def __init__(self):
        super(Cat2VecModel, self).__init__()
        self.model = BertModel.from_pretrained("setu4993/LaBSE")
        
    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = F.normalize((x[:, 1:, :]*mask[:, 1:, None]).mean(axis=1))
        return x
    


def inference(ds,model):
    BS = 256
    NW = 0
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
    tbar = tqdm(loader, file=sys.stdout)
    
    vs = []
    with torch.no_grad():
        for idx, (ids, masks) in enumerate(tbar):
            v = model(ids.cuda(), masks.cuda()).detach().cpu().numpy()
            vs.append(v)
    return np.concatenate(vs)      

def train():
    MAX_LEN = 64
    train = pd.read_csv("./data/train.csv")

    test=train.sample(n=5000)

    test['text'] = test[['name', 'categories']].fillna('').agg(' '.join, axis=1)#test['name'].map(str)+' ' +test['categories'].map(str)
    test['text'].drop_duplicates()
    tk=TokenizedDataset(test, MAX_LEN)
    

    cat2vec_model = Cat2VecModel()
    cat2vec_model = cat2vec_model.cuda()
    
    V = inference(tk,cat2vec_model)
    
    
    N = 5

    matcher = NearestNeighbors(n_neighbors=N, metric="cosine")
    matcher.fit(V)
    
    
    distances, indices = matcher.kneighbors(V)
    
    for i in range(1, N):
        test[f"match_{i}"] = test["text"].values[indices[:, i]]
        test[f"sim_{i}"] = np.clip(1 - distances[:, i], 0, None)
    
    return test
if __name__ == '__main__':
    test=train()
    test[['text','match_1','sim_1','match_2','sim_2']].to_csv('data/results.csv',index=False)
    g=test.groupby('point_of_interest')
    df=pd.DataFrame(columns=['text','match_1','sim_1','match_2','sim_2','point_of_interest'])
    Index=g.size()[g.size()>1].index
    for i in Index:
        print(i)
        df=pd.concat([test[test['point_of_interest']==i][['text','match_1','sim_1','match_2','sim_2','point_of_interest']],df],ignore_index=True)
    df.to_csv('data/results_paired.csv',index=False)