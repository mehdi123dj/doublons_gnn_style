
import pickle
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
from sklearn.decomposition import PCA


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
    NW = 2
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
    groups=g.groups

    for elem in groups:
        if len(groups[elem])>1:
            for i in range(1,len(groups[elem])):
                V_new[groups[elem][i]]=V_new[groups[elem][0]]

    print(len(V_new))
    return V_new

def Embed_transformers(df,destination_dir):
    
    MAX_LEN = 512 
    test = df
    test['text'] = test[['name', 'categories']].fillna('').agg(' '.join, axis=1)
    test=test.reset_index()
    test_drop=test.drop_duplicates(subset=['text'])

    tk=TokenizedDataset(test_drop, MAX_LEN)
    text2vec_model = Text2VecModel()
    text2vec_model = text2vec_model.cuda()

    V = inference(tk,text2vec_model)
    print(len(V))
    V=match(test, test_drop, V)
    print("match done")

    Ids=list(test['id'])
    ID_file = open(os.path.join(destination_dir,'ID.pkl'),"wb")
    pickle.dump(Ids,ID_file)

    Emb_file = open(os.path.join(destination_dir,'Transformers_embeddings.pkl'),"wb")
    pickle.dump(V,Emb_file)


