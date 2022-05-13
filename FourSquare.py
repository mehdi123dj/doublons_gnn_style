# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:05:47 2022

@author: remit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import haversine_distances


#%%
train = pd.read_csv("./data/train.csv")
print(train.head())


poi2distances = {}
for poi, df in tqdm(train[["latitude", "longitude", "point_of_interest"]].groupby("point_of_interest"),
                    total=train["point_of_interest"].nunique()):
    if len(df) == 1:
        # no matches
        continue
        
    distances = []
    distances_mat = haversine_distances(df[["latitude", "longitude"]].values)
    for i in range(len(df)):
        for j in range(len(df)):
            if j >= i:
                continue
            # haversine distance -> meters
            distances.append(distances_mat[i, j] * 6371000)
    poi2distances[poi] = distances
#%%
poi2distances_df = pd.DataFrame({
    "point_of_interest": list(poi2distances.keys()),
    "distances": list(poi2distances.values())
})
#%%
p=[len(x) for x in poi2distances_df['distances']]
print(max(p))
print(poi2distances_df['distances'].loc[np.array(p)>2000])
plt.hist(p, bins=200)
plt.show()

#%%
hist, bins = np.histogram(p, bins=500)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(p, bins=logbins)
plt.xscale('log')

#%%

print(poi2distances_df["distances"][0:5])

maxi={}
for index,row in poi2distances_df.iterrows():
    maxi[row['point_of_interest']]=max(row['distances'])

maxi_df= pd.DataFrame({
    "poi":list(maxi.keys()),
    "maxi":list(maxi.values())
    }
)


#%%
print(len(maxi_df["maxi"]))
print(len(maxi_df["maxi"].loc[maxi_df["maxi"]<100000]))
plt.hist(maxi_df["maxi"].loc[maxi_df["maxi"]<100000])
# print(max(maxi_df["maxi"]))

#%%
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")

train = pd.read_csv("./data/train.csv")

tokenize=[]
for i in train["categories"]:
    # print(type(i))
    token=tokenizer(str(i))
    tokenize.append(token)
df=train
df['Tokenize']=tokenize

#%%
from transformers import DistilBertModel, DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


text=train["name"][2]
encoding = tokenizer.encode_plus(text,
                                 # add_special_tokens = True,
                                 truncation = True,)
                                 # padding = "max_length",
                                 # return_attention_mask = True)#, return_tensors = "pt")
                                 


#%%
tokenizer =  BertTokenizerFast.from_pretrained("setu4993/LaBSE")
model =  BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()

test=train.sample(n=10)
test['all'] = test['name'].map(str)+' ' +test['categories'].map(str)
test['all'].drop_duplicates()
print(len(test))
# embedded_name=[]
# embedded_category=[]
embedded=[]
start=time.time()
for i in test.iterrows():
    # inputs_name=tokenizer(str(i[1]["name"]), return_tensors="pt", padding="max_length",max_length=MAX_LEN)
    # inputs_category=tokenizer(str(i[1]["categories"]), return_tensors="pt", padding="max_length",max_length=MAX_LEN)
    # inputs=tokenizer(str(i[1]["name"])+str(i[1]["categories"]), return_tensors="pt", padding="max_length",max_length=MAX_LEN)
    inputs=tokenizer.encode_plus(str(i[1]["all"]),
                                 padding="max_length",
                                 max_length=MAX_LEN,
                                 add_special_tokens=True,
                                 return_token_type_ids=True,
                                 truncation=True)
    ids = torch.LongTensor(inputs['input_ids'])
    mask = torch.LongTensor(inputs['attention_mask'])
    
    cat.append()
#%%
import torch
import time
from transformers import  BertModel,  BertTokenizerFast

MAX_LEN = 64
train = pd.read_csv("./data/train.csv")
import sys
from torch.utils.data import DataLoader, Dataset
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
                                            max_length=MAX_LEN,
                                            add_special_tokens=True,
                                            return_token_type_ids=True,
                                            truncation=True)
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])
        return ids,mask
    
test=train.sample(n=100)
test['text'] = test['name'].map(str)+' ' +test['categories'].map(str)
test['text'].drop_duplicates()
tk=TokenizedDataset(test, MAX_LEN)

#%%
BS = 256
NW = 4    
model =  BertModel.from_pretrained("setu4993/LaBSE")
model = model.eval()


def inference(ds,model):
    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
    tbar = tqdm(loader, file=sys.stdout)
    
    vs = []
    with torch.no_grad():
        for idx, (ids, masks) in enumerate(tbar):
            v = model(ids.cuda(), masks.cuda()).detach().cpu().numpy()
            vs.append(v)
    return np.concatenate(vs)


V = inference(tk,model)
V.shape


# #%%
 
#     with torch.no_grad():
#         # outputs_name = model(**inputs_name)
#         # outputs_category = model(**inputs_category)
#         # outputs = model(**inputs)
#         outputs = model(ids,mask)[0]
#     # embeddings_name =outputs_name.pooler_output
#     # embedded_name.append(embeddings_name)
#     # embeddings_category =outputs_category.pooler_output
#     # embedded_category.append(embeddings_category)    
#     embeddings=outputs.pooler_output
#     embedded.append(embeddings)    
    
# end=time.time()
# print(end-start)
# df=test
# # df["name_embedded"]=embedded_name
# # df["category_embedded"]=embedded_category
# df["embedded"]=embedded
# # english_sentences = [
# #     "dog",
# #     "Puppies are nice.",
# #     "I enjoy taking long walks along the beach with my dog.",
# # ]
# # english_inputs = tokenizer(english_sentences, return_tensors="pt")#, padding=True)

# # with torch.no_grad():
# #     english_outputs = model(**english_inputs)
    
# # italian_sentences = [
# #     "cane",
# #     "I cuccioli sono carini.",
# #     "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.",
# # ]
# # japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]
# # italian_inputs = tokenizer(italian_sentences, return_tensors="pt")#, padding=True)
# # japanese_inputs = tokenizer(japanese_sentences, return_tensors="pt")#, padding=True)

# # with torch.no_grad():
# #     italian_outputs = model(**italian_inputs)
# #     japanese_outputs = model(**japanese_inputs)

# # english_embeddings = english_outputs.pooler_output
# # italian_embeddings = italian_outputs.pooler_output
# # japanese_embeddings = japanese_outputs.pooler_output
# # import torch.nn.functional as F


# # def similarity(embeddings_1, embeddings_2):
# #     normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
# #     normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
# #     return torch.matmul(
# #         normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
# #     )


# # print(similarity(english_embeddings, italian_embeddings))
# # print(similarity(english_embeddings, japanese_embeddings))
# # print(similarity(italian_embeddings, japanese_embeddings))

# #%%
# df=train
# train[["name","categories"]]
# df['all'] = df['name'].map(str)+' ' +df['categories'].map(str)
