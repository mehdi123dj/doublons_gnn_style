# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:29:37 2022

@author: remit
"""

import torch
import pickle 
import json
import numpy as np
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url,Data


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data,self.slices,self.sizes = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['preProcess600000true_edges.pkl','preProcess600000kemb=10kspat=10.json','preProcess600000representation.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        L=[]
        with open(self.raw_paths[0],'rb') as file:
            edge_index_true = pickle.load(file)
        
        with open(self.raw_paths[1],'r') as file:
            preProcess = json.load(file)
        
        with open(self.raw_paths[2],'rb') as file:
            nodes_features = pickle.load(file)
        # i=0
        edge_true=set(zip(edge_index_true[0],edge_index_true[1]))
        # print(edge_true)
        keys=list(preProcess.keys())
        for elem in preProcess:
            #print(elem)
            edge = preProcess[elem]["representation_indices"]
            edge_initial = preProcess[elem]["initial_indices"]
            
            edge_label = []
            node_feature = nodes_features[elem]
            #if node_feature.shape[0]<1000:
                #print(node_feature.size())
                #continue
                
            #print(elem)
            for item in set(zip(edge_initial[0],edge_initial[1])):
                
                if item in edge_true:
                    edge_label.append(1)
                else:
                    edge_label.append(0)
            # print(edge)
            
            edge_label = torch.tensor(edge_label)
            edge_index = torch.tensor(edge).long()
            edge_initial_index = torch.tensor(edge_initial)
            node_feature = torch.tensor(node_feature)
           
            data = Data(x = node_feature,
                        edge_index = edge_index,
                        edge_initial_index = edge_initial_index,
                        edge_label = edge_label,
                        country = torch.tensor([keys.index(elem)])
                        )
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
     
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            
            L.append(data)

        self.data, self.slices = self.collate(L)
        torch.save((self.data, self.slices,len(L)), self.processed_paths[0])
        
    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx):
    #     """ - Equivalent to __getitem__ in pytorch
    #         - Is not needed for PyG's InMemoryDataset
    #     """
    #     if self.test:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_test_{idx}.pt'))
    #     else:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_{idx}.pt'))   
    #     return data