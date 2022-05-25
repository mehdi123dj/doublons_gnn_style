# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:25:21 2022

@author: remit
"""

import json
import pandas as pd 
import pickle
import time
import torch
from torch_cluster import knn_graph
from sklearn.metrics.pairwise import haversine_distances as dist
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
import numpy as np 
from math import radians, cos, sin, asin, sqrt



class Graph():
    
    def __init__(self,ids_path,embedding_path,data_path):
        with open(embedding_path,'rb') as file:
            self.emmbedding = pickle.load(file)

        with open(ids_path,'rb') as file:
            self.ids = pickle.load(file)

        data = pd.read_csv(data_path)
        self.df = data[data['id'].isin(self.ids)]
        self.df = self.df.reset_index(drop=True)
        
    def true_edges(self):


        # Version ordered by country for efficiency and computability
        # groups_country = self.df.groupby(['country']).groups
        # groups={}
        # repeated={}
        # indexes={}
        # edges_true={}
        # for elem in groups_country:
        #     groups[elem] = self.df.iloc[groups_country[elem]].groupby(["point_of_interest"]).groups
        #     repeated[elem] = {p:groups[elem][p] for p in groups[elem] if len(groups[elem][p])>1}
        #     indexes[elem] = {repeated[elem][p][0] : list(set(repeated[elem][p][1:])) for p in repeated[elem]}
        #     edges_true[elem] = [(u,v)  for u in  indexes[elem] for v in indexes[elem][u]]
        # U=list(edges_true.values())
        # K=[item for elem in U for item in elem]
        # B = set(edgify(K))

        """
        Number of unique true links ordered by country is 397808 for the whole dataframe

        On first 400000 rows got 60478
        """

        # Version without countrying

        groups = self.df.groupby(["point_of_interest"]).groups
        repeated = {p : groups[p] for p in groups if len(groups[p])>1}
        indexes = {repeated[p][0] : list(set(repeated[p][1:])) for p in repeated}
        edges_true = [(u,v)  for u in  indexes for v in indexes[u]]
        B = set(self.edgify(edges_true))

        """
        Number of unique true links ordered by country is 398840 for the whole dataframe

        Number of unique true links ordered by country is 60725 for 400000 first values of the dataframe

        -> If we choose to saerch by country we loose 0.00259% of possible links we seems to be okk for the 
        increase value in knn graph function due to smaller sets
        """
        print("processed")
        return B
        
    def creation(self,k_emb_list,k_spat_list):
        B = self.true_edges()
        sort_index=np.argsort(self.ids)
        embedding_new=np.array(self.emmbedding)[sort_index]
        groups_country = self.df.groupby(['country']).groups
        
        L_A={}
        Repre={}
        for k_emb in k_emb_list:
            for k_spat in k_spat_list:
                Q=set()
                for elem in groups_country:
                    print(elem)
                    A=set()
                    rep=set()
                    L_A[elem]={}
                    embedding_temp=embedding_new[np.array(groups_country[elem])]
                    N=len(embedding_temp)
                    if N<=1:
                        T1_emb,T2_emb = [],[]
                    elif k_emb < N:
                        A_emb = kneighbors_graph(embedding_temp,n_neighbors = k_emb,metric = 'cosine')
                        T1_emb,T2_emb =A_emb.nonzero()
                    else:
                        A_emb = kneighbors_graph(embedding_temp,n_neighbors = N-1,metric = 'cosine')
                        T1_emb,T2_emb=A_emb.nonzero()
                    edges_initial_emb = [(groups_country[elem][u],groups_country[elem][v]) for u,v in zip(T1_emb,T2_emb)]
                    Emb_initial=set(self.edgify(edges_initial_emb))
                    
                    edges_representation_emb = [(u,v) for u,v in zip(T1_emb,T2_emb)]
                    Emb_representation=set(self.edgify(edges_representation_emb))

                    if N<=1:
                        T1_spat,T2_spat = [],[]
                    elif k_spat < N:
                        
                        A_spat = kneighbors_graph([[radians(item[0]),radians(item[1])] for item in np.array(self.df[["latitude","longitude"]].iloc[groups_country[elem]])],
                                                  n_neighbors = k_spat,
                                                  metric = 'haversine')
                        
                        T1_spat,T2_spat = A_spat.nonzero()
                    else:
                        A_spat = kneighbors_graph([[radians(item[0]),radians(item[1])] for item in np.array(self.df[["latitude","longitude"]].iloc[groups_country[elem]])],
                                                  n_neighbors = N-1,
                                                  metric = 'haversine')
                        
                        T1_spat,T2_spat = A_spat.nonzero()
                    edges_initial_spat = [(groups_country[elem][u],groups_country[elem][v]) for u,v in zip(T1_spat,T2_spat)]
                    Spat_initial=set(self.edgify(edges_initial_spat))
                    Temp_initial = Spat_initial.union(Emb_initial)
                    A=Temp_initial
                    Q=Q.union(A)
                    
                    edges_representation_spat = [(u,v) for u,v in zip(T1_spat,T2_spat)]
                    Spat_representation = set(self.edgify(edges_representation_spat))
                    Temp_representation = Spat_representation.union(Emb_representation)
                    rep=Temp_representation

                    
                    X,Y=[],[]
                    for a,b in A:
                        X.append(int(a))
                        Y.append(int(b))
                    L_A[elem]['initial_indices']=[X,Y]
                    
                    X,Y=[],[]
                    for a,b in rep:
                        X.append(int(a))
                        Y.append(int(b))
                    L_A[elem]['representation_indices']=[X,Y]
                    Repre[elem]=embedding_temp
                print("percentage of overlapping with initial set :",self.overlap(Q,B),'Jaccard indice:',self.Jaccard(Q, B)," with k_emb=",k_emb," and k_spat=",k_spat)
        
        
        representation_file = open('./data/raw/preProcess'+str(len(self.df))+'representation.pkl',"wb")
        pickle.dump(Repre,representation_file)
        
        
        edge_file = open('./data/raw/preProcess'+str(len(self.df))+'kemb='+str(k_emb)+'kspat='+str(k_spat)+'.json',"w")
        json.dump(L_A,edge_file)
        
        X,Y=[],[]
        for a,b in B:
            X.append(a)
            Y.append(b)
        edge_true=[X,Y]
        
        true_edge_file = open('./data/raw/preProcess'+str(len(self.df))+'true_edges.pkl',"wb")
        pickle.dump(edge_true,true_edge_file)
        return L_A
    
        
    def edgify(self,l):
        return [(min([u,v]),max([u,v])) for u,v in l]
    def overlap(self,A,B):
        return len(A.intersection(B))/min([len(A),len(B)])
    def Jaccard(self,A,B):
        return len(A.intersection(B))/len(A.union(B))


if __name__ == '__main__':
    start=time.time()
    graph=Graph('/kaggle/input/four-square-competition-text-embedding-600000-rows/IDKaggle600000.pkl', '/kaggle/input/four-square-competition-text-embedding-600000-rows/EmbeddingsKaggle600000.pkl', "/kaggle/input/foursquare-location-matching/train.csv")
    graph.creation([10],[10])
    end=time.time()
    print(end-start)   

