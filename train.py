# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:26:58 2022

@author: remit
"""

import torch 

import os 
from torch_geometric.loader import DataLoader
import copy 
from model import train_link_classifier, test_link_classifier , GNN_link_classifier 
from PersoDataset import MyOwnDataset
import argparse


data_dir = "./data"
models_dir = "./model"


def main(): 

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
    )

    parser.add_argument(
        "-wd",
        "--weight-decay",
        default=0.,
        type=float,
    )

    parser.add_argument(
        "-sp",
        "--save-path",
        default=models_dir,
        type=str,
    )

    parser.add_argument(
        "-mn",
        "--model-name",
        default="batched_inductive_link_class.mdl",
        type=str,
    )

    parser.add_argument(
        "-fp",
        "--data-fold",
        default= data_dir,
        type=str,
    )


    parser.add_argument(
        "-e",
        "--number-epochs",
        default=200,
        type=int,
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2)

    args = parser.parse_args()

    ##########################################################################################################################################
    ##########################################################################################################################################
    
    dataset=MyOwnDataset(data_dir)
    print(len(dataset))
    train_dataset = dataset[len(dataset)//8:]
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    test_dataset = dataset[:len(dataset)//8]
    test_loader = DataLoader(test_dataset, args.batch_size)
    
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    SAVEPATH = os.path.join(args.save_path,args.model_name)
    nepochs = args.number_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    ##########################################################################################################################################
    ##########################################################################################################################################

    HCs = {0:256,1:128, 2:64}
    model = GNN_link_classifier(hidden_channels=HCs, out_channels=16,num_layers = 3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)#, weight_decay = weight_decay)

    ##########################################################################################################################################
    ##########################################################################################################################################
    best_test_auc = final_test_auc = 0
    for epoch in range(1, nepochs+1):
        loss = train_link_classifier(model,train_loader,optimizer,device)
        #val_auc,val_rec,val_prec = test_link_classifier()
        test_auc,test_rec,test_prec = test_link_classifier(model,test_loader,device)
        if test_auc > best_test_auc:
            best_model = copy.deepcopy(model)
            best_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}'
            f'Test Rec: {test_rec:.4f}, Test Prec: {test_prec:.4f},')
    print(f'Final Test: {best_test_auc:.4f}')
    torch.save(best_model,SAVEPATH)

    ##########################################################################################################################################
    ##########################################################################################################################################

if __name__ == "__main__":
    main()