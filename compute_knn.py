
import pandas as pd 
import gc 
import numpy as np 
import pickle 
from sklearn.neighbors import NearestNeighbors
import os 

def dist_to_edges(l_dist,l_idx,s,my_map):    
    filtred_l = np.where(l_dist<=s)
    edges = []
    for i in range(len(filtred_l[0])):
        u,v = filtred_l[0][i],filtred_l[1][i]
        #print(u,v)
        U,V = my_map[u],my_map[l_idx[u][v]]
        edges.append((U,V))
    return [(min(u,v),max(u,v)) for u,v in edges if u!=v]

def get_identiques(groups):
    identiques_pays = []
    for d in groups:
        identiques_pays.extend(list(combinations(list(groups[d]), 2)))
    identiques_pays = edgify(identiques_pays)
    return identiques_pays

def overlap(A,B):
    return len(A.intersection(B))/min([len(A),len(B)])

def edgify(l):
    return list(set([(min([u,v]),max([u,v])) for u,v in l if u!=v]))

def compute_Knns(df,E,T,k_dist,k_w2v,k_trans,dir_path)
    group_cols = ['name','address']

    doublons_dict = dict()

    dist_dist_dict = dict()
    dist_idx_dict = dict()

    w2v_dist_dict = dict()
    w2v_idx_dict = dict()

    trans_dist_dict = dict()
    trans_idx_dict = dict()

    for pays in set(df['country'].dropna()):
        
        print(pays)
        df_pays = df[df["country"]== pays]

        X_dist = np.array(df_pays[["latitude","longitude"]])
        X_w2v= E[df_pays.index]
        X_trans = T[df_pays.index]
        
        if X_dist.shape[0]>k_dist:
            neigh_dist = NearestNeighbors(n_neighbors=k_dist, algorithm='auto', metric='haversine', n_jobs=-1).fit(X_dist)
        else: 
            neigh_dist = NearestNeighbors(n_neighbors = X_dist.shape[0], algorithm='auto', metric='haversine', n_jobs=-1).fit(X_dist)
        dist_dist,dist_idx  = neigh_dist.kneighbors(X_dist)
        dist_dist_dict[pays] = dist_dist
        dist_idx_dict[pays] = dist_idx
        del dist_dist,dist_idx,neigh_dist,X_dist
        gc.collect()


        if X_w2v.shape[0]>k_w2v:
            neigh_w2v = NearestNeighbors(n_neighbors=k_w2v, algorithm='auto', metric='cosine', n_jobs=-1).fit(X_w2v)
        else: 
            neigh_w2v = NearestNeighbors(n_neighbors = X_w2v.shape[0], algorithm='auto', metric='cosine', n_jobs=-1).fit(X_w2v)
        w2v_dist,w2v_idx  = neigh_w2v.kneighbors(X_w2v)
        w2v_dist_dict[pays] = w2v_dist
        w2v_idx_dict[pays] = w2v_idx
        del w2v_dist,w2v_idx,neigh_w2v,X_w2v
        gc.collect()
       
        if X_trans.shape[0]>k_trans:
            neigh_trans = NearestNeighbors(n_neighbors=k_trans, algorithm='auto', metric='cosine', n_jobs=-1).fit(X_trans)
        else:
            neigh_trans = NearestNeighbors(n_neighbors=X_trans.shape[0], algorithm='auto', metric='cosine', n_jobs=-1).fit(X_trans)
        trans_dist,trans_idx  = neigh_trans.kneighbors(X_trans)
        trans_dist_dict[pays] = trans_dist
        trans_idx_dict[pays] = trans_idx
        del trans_dist,trans_idx,neigh_trans,X_trans
        gc.collect()

    file = os.path.join(dir_path,'dist_dist_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(dist_dist_dict,dict_file)
    file = os.path.join(dir_path,'dist_idx_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(dist_idx_dict,dict_file)

    file = os.path.join(dir_path,'w2v_dist_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(w2v_dist_dict,dict_file)
    file = os.path.join(dir_path,'w2v_idx_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(w2v_idx_dict,dict_file)

    file = os.path.join(dir_path,'trans_dist_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(trans_dist_dict,dict_file)
    file = os.path.join(dir_path,'trans_idx_dict.pkl')
    with open(file,"wb") as dict_file:
        pickle.dump(trans_idx_dict,dict_file)



def compute_overlap(df,dist_dist_dict,dist_idx_dict,
                    w2v_dist_dict,w2v_idx_dict,
                    trans_dist_dict,trans_idx_dict,
                    k_dist,k_w2v,k_trans,
                    data_dir):
    group_cols = ['name','address']

    w2vec = []
    transformers = []
    spatial = []

    identiques = []
    real_edges  = []


    for pays in set(df['country'].dropna()):
        
        print(pays)
        df_pays = df[df["country"]== pays]
        doublons = df_pays.groupby("point_of_interest").groups
        doublons_pays = {p : doublons[p] for p in doublons if len(doublons[p])>1}

        dist_dist = dist_dist_dict[pays][:,:k_dist]
        dist_idx = dist_idx_dict[pays][:,:k_dist]
        w2v_dist = w2v_dist_dict[pays][:,:k_w2v]
        w2v_idx  = w2v_idx_dict[pays][:,:k_w2v]
        trans_dist = trans_dist_dict[pays][:,:k_trans]
        trans_idx  = trans_idx_dict[pays][:,:k_trans]


        #########################################################################
        s_w2v = np.max(w2v_dist)
        s_trans = np.max(trans_dist)
        s_dist = np.max(dist_dist)

        group_cols = ['name','address']

        idxs = list(df_pays.index)
        my_map = {i  : idxs[i] for i in range(len(idxs))}
        

        edges_dist = dist_to_edges(dist_dist,dist_idx,s_dist,my_map)
        edges_w2v = dist_to_edges(w2v_dist,w2v_idx,s_w2v,my_map)
        edges_trans = dist_to_edges(trans_dist,trans_idx,s_trans,my_map)

        w2vec.extend(edges_w2v)
        transformers.extend(edges_trans)
        spatial.extend(edges_dist)
        
        for d in doublons_pays:
            real_edges.extend(idx_to_edge(doublons_pays[d]))

        groups = df_pays.groupby(group_cols).groups
        groups = {g : groups[g] for g in groups if len(groups[g])>1}
        identiques.extend(get_identiques(groups))


    edges = list(set(w2vec+transformers+spatial))
    all_edges = list(set(edges+identiques))

    graph_edges = list(set(all_edges+real_edges))

    false_edges = list(set(graph_edges)-set(real_edges))
    labeled_graph_edges = []
    for u,v in false_edges : 
        labeled_graph_edges.append((u,v,0))
    for u,v in real_edges : 
        labeled_graph_edges.append((u,v,1))
    file = os.path.join(data_dir,'labled_edges.pkl')
    with open(file,"wb") as list_file:
        pickle.dump(labeled_graph_edges,list_file)

    print("w2v_overlap : ", overlap(set(real_edges),set(w2vec)))
    print("trans_overlap : ", overlap(set(real_edges),set(transformers)))
    print("dis_overlap : ", overlap(set(real_edges),set(spatial)))
    print("sum_overlap : ", overlap(set(real_edges),set(edges)))
    print("added_identiques_overlap : ",overlap(set(real_edges),set(all_edges)))
    print(' ')
    print("there are",len(real_edges),' real doubles ')
    print("there are",len(edges),' susceptible doubles ')
    print("there are",len(all_edges),' susceptible doubles when adding identicals ')

