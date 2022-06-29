

from model import Encoder, train , corruption, summary 
from Batch_dataset import MyOwnDataset, get 
import torch 
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.loader.dataloader import DataLoader
from typing import List
import os 
import copy
import argparse
import numpy as np 

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
        "-sp",
        "--save-path",
        default='./models',
        type=str,
    )

    parser.add_argument(
        "-sbs",
        "--sample-bs",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "-tbs",
        "--train-bs",
        default=2,
        type=int,
    )


    parser.add_argument(
        "-kc",
        "--k-couches",
        default=3,
        type=int,
    )

    parser.add_argument(
        "-ns",
        "--num-samples",
        default=10000,
        type=int,
    )

    parser.add_argument(
        "-nc",
        "--num-chunks",
        default=2,
        type=int,
    )

    parser.add_argument(
        "-ne",
        "--num-epochs",
        default=100,
        type=int,
    )

    parser.add_argument(
        "-hc",
        "--hidden-channels",
        default=512,
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


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #########################################################################################################
    ## parametre pour la creation du dataset, une fois celui-ci créé, il faudra aller le supprimer 
    ## avant d'espérer un changement par la modification de ces paramètres (si non on accède à la dernière copie par défaut)
    #########################################################################################################
    root = args.root
    sampling_BS = args.sample_bs
    k_kouches = args.k_couches
    num_neighbors = [-1]*k_kouches

    num_samples = args.num_samples
    n_chunks = args.num_chunks
    DS = MyOwnDataset(root, sampling_BS, num_neighbors, num_samples, n_chunks)
    #########################################################################################################
    ## parametres spécifiques à l'entrainement du modèle DeepGraphInfomax
    #########################################################################################################
    n_epoch = args.num_epochs 
    in_channels = 969
    hidden_channels = args.hidden_channels
    channels = [in_channels]+([hidden_channels]*k_kouches)
    train_BS = args.train_bs
    #########################################################################################################

    processed_dir = [os.path.join(root,"processed",u) for u in os.listdir(os.path.join(root,"processed"))]
    train_list = []
    for idx in range(DS.length()):
    #for idx in range(1):
        train_list.extend(get(processed_dir,idx))

    train_loader = DataLoader(train_list,batch_size = train_BS,shuffle=True)

    model = DeepGraphInfomax(
        hidden_channels = channels[-1],
        encoder=Encoder(channels),
        summary=summary,
        corruption=corruption
                            )
    
    setattr(model, 'sample_bs',sampling_BS)
    setattr(model, 'k_couches',k_kouches)
    setattr(model, 'num_neighbors',num_neighbors)
    setattr(model, 'num_samples',num_samples)
    setattr(model, 'n_chunks',n_chunks)


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    best_loss = np.inf 
    for epoch in range(n_epoch):
        print('epoch : {:d}'.format(epoch+1))
        loss = train(model,optimizer,epoch,train_loader)
        print('Loss:', loss)
        if loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = loss
            torch.save(best_model, os.path.join(args.save_path,'unsupervised.mdl'))
            print("[New best Model]")



if __name__ == "__main__":
    main()
