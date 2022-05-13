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
import os
import datetime
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
    
class Text2VecModel(nn.Module):
    def __init__(self):
        super(Text2VecModel, self).__init__()
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


def match(df,df_drop,V):
    V_new=[[] for i in range(len(df))]
    j=0
    for i in df_drop.index:
        V_new[i]=V[j]
        j+=1
    
    
    g=df.groupby('text')
    Index=g.size()[g.size()>1].index
    for elem in Index:
       L= df[df['text']==elem].index
       for i in L[1:]:

            V_new[i]=V_new[L[0]]
    
    
    print(len(V_new))
    return V_new

def train():
    MAX_LEN = 64
    
    # read csv file
    train = pd.read_csv("./data/train.csv")

    test=train.sample(n=20000)
    test['text'] = test[['name', 'categories']].fillna('').agg(' '.join, axis=1)
    test=test.reset_index()
    
    test_drop=test.drop_duplicates(subset=['text'])
    tk=TokenizedDataset(test_drop, MAX_LEN)

    text2vec_model = Text2VecModel()
    text2vec_model = text2vec_model.cuda()
    
    V = inference(tk,text2vec_model)
    print(len(V))
    V=match(test, test_drop, V)
    N = 4

    matcher = NearestNeighbors(n_neighbors=N, metric="cosine")
    matcher.fit(V)
    
    
    distances, indices = matcher.kneighbors(V)
    
    for i in range(1, N):
        test[f"match_{i}"] = test["text"].values[indices[:, i]]
        test[f"sim_{i}"] = np.clip(1 - distances[:, i], 0, None)
    
    Evaluate_and_Register(test,N)

    return test


def Evaluate_and_Register(df,N):
    
    columns=[]
    for i in range(1,N):
        columns.append("match_{}".format(i))
        columns.append("sim_{}".format(i))
        
    df[['id','text']+columns].to_csv(os.path.join('data','results',datetime.datetime.today().strftime('%m-%d-%H-%M')+'.csv'),index=False)
    g=df.groupby('point_of_interest')
    df_new=pd.DataFrame(columns=['point_of_interest','text']+columns)
    Index=g.size()[g.size()>1].index
    for i in Index:
        df_new=pd.concat([df[df['point_of_interest']==i][['point_of_interest','id','text']+columns],df_new],ignore_index=True)
    df_new.to_csv(os.path.join('data','results','duplicate'+datetime.datetime.today().strftime('%m-%d-%H-%M')+'.csv'),index=False)
    
    
if __name__ == '__main__':
    start=time.time()
    test=train()
    end=time.time()
    print(end-start)