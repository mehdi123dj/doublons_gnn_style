

from model import Encoder, train , corruption, summary , XGboost_classifier
from Batch_dataset import MyOwnDataset, get 
import torch 
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.loader.dataloader import DataLoader
from typing import List
import os 
import copy
import argparse
import numpy as np 
import gc 
def main():

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--root",
        default='./data',
        type=str,
    )

    parser.add_argument(
        "-mp",
        "--model-path",
        default='./models',
        type=str,
    )

    parser.add_argument(
        "-tbs",
        "--train-bs",
        default=2,
        type=int,
    )


    parser.add_argument(
        "-estim",
        "--n-estim",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "-tr",
        "--train-ratio",
        default=.8,
        type=float,
    )


    parser.add_argument(
        "-dr",
        "--depth-ratio",
        default=.05,
        type=float,
    )

    parser.add_argument(
        "-tm",
        "--tree-method",
        default='auto',
        type=str,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=.01,
        type=float,
    )


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #########################################################################################################
    #########################################################################################################
    root = args.root
    train_BS  = args.train_bs
    train_ratio = args.train_ratio
    depth_ratio = args.depth_ratio
    tree_method = args.tree_method #'auto'
    learning_rate = args.learning_rate 
    n_estimators = args.n_estim
    #########################################################################################################
    #########################################################################################################
    GRM = torch.load(os.path.join(args.model_path,'unsupervised.mdl')).to(device)

    sampling_BS = GRM.sample_bs
    num_neighbors = GRM.num_neighbors
    num_samples = GRM.num_samples
    n_chunks = GRM.n_chunks

    DS = MyOwnDataset(root, sampling_BS, num_neighbors, num_samples, n_chunks)

    processed_dir = [os.path.join(root,"processed",u) for u in os.listdir(os.path.join(root,"processed"))]
    train_list = []
    for idx in range(DS.length()):
        train_list.extend(get(processed_dir,idx))

    data_loader = DataLoader(train_list,batch_size = train_BS,shuffle=True)
    GRM = torch.load(os.path.join(args.model_path,'unsupervised.mdl')).to(device)
    max_depth = int(train_list[0].x.shape[1]*depth_ratio)
    classifier = XGboost_classifier(GRM,data_loader,train_ratio,max_depth,tree_method,learning_rate,n_estimators)
    del data_loader,GRM, DS, train_list
    gc.collect()
    X,Y,Z = classifier.embed()
    recall_1, precision_1, recall_2, precision_2, recall_3, precision_3 = classifier.train_classifiers(X,Y,Z)
    
    print(recall_1, precision_1)
    print(recall_2, precision_2)
    print(recall_3, precision_3)


if __name__ == "__main__":
    main()
