import torch
import pickle 
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url,Data
from torch_geometric.loader import DataLoader,RandomNodeSampler,NeighborLoader
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from torch_geometric.transforms import LargestConnectedComponents,RandomNodeSplit,LineGraph
from math import radians, cos, sin, asin, sqrt
import pandas as pd 
import gc 
import random 
import copy 
import os 
import itertools
import scipy.sparse as sp
import os.path as osp

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lon2)
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

def compose_new_x(X,trace,sample_map):
    T=torch.zeros(trace.edge_index.shape[1],X.shape[1],dtype = X.dtype)
    for i in range(trace.edge_index.shape[1]):
        [u,v] = trace.edge_index[:,i].tolist()
        u,v = sample_map[u],sample_map[v]
        T[i] += torch.tensor((X[u]-X[v]))
    return T

def loader_to_list(data,slices,idx_map):
    l = []
    for i in range(slices["n_id"].shape[0]-1):
        I_edge,J_edge = slices["edge_index"][i].item(),slices["edge_index"][i+1].item()
        I_node,J_node = slices["n_id"][i].item(),slices["n_id"][i+1].item()
        x_i = data.x[I_node:J_node]
        y_i = data.y[I_node:J_node]
        edge_index_i = data.edge_index[:,I_edge:J_edge]
        idx_i0 = torch.tensor([idx_map[u][0] for u in data.n_id[I_node:J_node].tolist()])
        idx_i1 = torch.tensor([idx_map[u][1] for u in data.n_id[I_node:J_node].tolist()])
        d = Data(
            x = x_i,
            y = y_i,
            edge_index = edge_index_i,
            edge_original  = torch.stack((idx_i0,idx_i1),dim=0)
            )
        l.append(d)
    return l 


def process_chunk(data,
                X,chunk,df,
                BS,num_neighbors,
                num_samples,
                ):

    sample = list(chunk)
    sample_map = {s : i for i,s in enumerate(sample)}
    sub_graph_edge_index,sub_graph_edge_label = subgraph(subset = sample, edge_index = data.edge_index,
                                    edge_attr = data.edge_label ,relabel_nodes= False)
    new_data =  Data(
                edge_index = sub_graph_edge_index,
                edge_label = sub_graph_edge_label,
                )
    idx_map = {i : (new_data.edge_index[0][i].item(),new_data.edge_index[1][i].item()) for i in range(new_data.edge_index.shape[1])}

    trace = copy.copy(new_data)
    liner = LineGraph()
    line_G = liner(new_data)
    line_G.y = line_G.edge_label
    line_G.n_id = torch.arange(line_G.num_nodes)


    delattr(line_G, 'edge_label')
    delattr(line_G, 'num_nodes')

    X_new = compose_new_x(X,trace,sample_map)
    to_map  = [(trace.edge_index[0][i].item(),trace.edge_index[1][i].item()) for i in range(trace.edge_index.shape[1])]
    to_map = {i : e for i,e in enumerate(to_map)}
    X_ = np.array(df.iloc[[u for u,v in to_map.values()]][["longitude","latitude"]])
    Y = np.array(df.iloc[[v for u,v in to_map.values()]][["longitude","latitude"]])
    D = haversine(X_[:,0], X_[:,1], Y[:,0], Y[:,1])
    X = torch.tensor(np.column_stack((np.array(X_new),np.array(D))),dtype = torch.float16)
    line_G.x = X
    del to_map, X_,Y, X, D
    gc.collect()

    line_G.num_nodes = line_G.x.shape[0]
    adj = to_scipy_sparse_matrix(line_G.edge_index, num_nodes=line_G.num_nodes)
    num_components, component = sp.csgraph.connected_components(adj)
    compo_connexes = pd.DataFrame.from_dict({"compo" : component}).groupby("compo").groups
    pioche = torch.tensor(list(itertools.chain(*[list(v) for k,v in compo_connexes.items() if len(v)>len(num_neighbors)])))
    loader =  NeighborLoader(line_G, input_nodes = pioche, 
                            batch_size=BS, num_neighbors=num_neighbors,
                            shuffle = True
                            )
    return loader,idx_map

def get(processed_dir,idx):

    loader_path  = sorted([u for u in processed_dir if u[-2:] == 'pt'])
    map_path  = sorted([u for u in processed_dir if u[-3:] == 'map'])

    loader = torch.load(loader_path[idx])
    with open(map_path[idx], 'rb') as handle:
        idx_map = pickle.load(handle)
    data_list = []
    for mini_batch in loader:
        idx_i0 = torch.tensor([idx_map[u][0] for u in mini_batch.n_id.tolist()])
        idx_i1 = torch.tensor([idx_map[u][1] for u in mini_batch.n_id.tolist()])
        setattr(mini_batch,"edge_original",torch.stack((idx_i0,idx_i1),dim=0))
        #delattr(mini_batch, 'n_id')
        delattr(mini_batch, 'num_nodes')
        data_list.append(mini_batch)
    return data_list


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, BS, num_neighbors,num_samples,n_chunks):
        self.BS  = BS
        self.num_neighbors = num_neighbors
        self.num_samples = num_samples
        self.n_chunks = n_chunks
        super().__init__(root)

    @property 
    def raw_file_names(self):
        return ['labled_edges.pkl','ID600k.pkl',"Transformers_embeddings_600k.pkl",'W2V_Embeddings.pkl','train.csv']

    @property
    def processed_file_names(self):
        return ['data_{:d}.pt'.format(i) for i in range(self.n_chunks)]

    def process(self):

        with open(self.raw_paths[0],'rb') as file:
            labled_edges = pickle.load(file)
        edge_labels  = np.array([c for u,v,c in labled_edges])
        all_edges_from  = torch.tensor([u for u,v,c in labled_edges])
        all_edges_to  = torch.tensor([v for u,v,c in labled_edges])
        positive_idx = np.where(edge_labels == 1)
        positive_idx = np.where(edge_labels == 0)
        edge_label = torch.tensor(edge_labels)
        del labled_edges
        gc.collect()

        data = Data(
                    edge_index = torch.stack((all_edges_from,all_edges_to),dim = 0).long(),
                    edge_label = edge_label,
                    num_nodes = 600000
                    )
        
        nodes = set(data.edge_index[0].tolist()).union(set(data.edge_index[1].tolist()))
        del all_edges_from, all_edges_to, edge_label
        gc.collect()

        with open(self.raw_paths[1],'rb') as file:
            IDs = pickle.load(file)
        sort_index = np.argsort(IDs)
        with open(self.raw_paths[2],'rb') as file:
            T = pickle.load(file)
        T = np.array(T)
        T = T[sort_index]
        with open(self.raw_paths[3],'rb') as file:
            E = pickle.load(file)
        node_features_emb = torch.cat((torch.tensor(T,dtype = torch.float16),
                                    torch.tensor(E,dtype = torch.float16)),dim=1)

        del E,T
        gc.collect()
        X = node_features_emb
        del node_features_emb
        gc.collect()

        DF  =  pd.read_csv(self.raw_paths[4])
        df = DF[DF['id'].isin(IDs)]
        df = df.reset_index(drop=True)

        shuffled_idx = np.array([i for i in range(len(df))])
        random.shuffle(shuffled_idx)
        chunk_list = np.array_split(shuffled_idx, self.n_chunks)
        #data_list = []
        for i,chunk in enumerate(chunk_list):
            print(i)
            loader,idx_map = process_chunk(data,X,chunk,df,self.BS,self.num_neighbors,self.num_samples)
            torch.save(loader, self.processed_paths[i])
        
            with open(self.processed_paths[i][:-2]+'map', 'wb') as handle:
                pickle.dump(idx_map, handle)
            del loader, idx_map
            gc.collect()

    def length(self):
        return len(self.processed_paths)
