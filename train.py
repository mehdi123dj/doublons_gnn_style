

from model import Encoder, train, test , corruption
from Batch_dataset import MyOwnDataset, get 
import torch 
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.loader.dataloader import DataLoader
from typing import List
import os 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################################################################
## parametre pour la creation du dataset, une fois celui-ci créé, il faudra aller le supprimer 
## avant d'espérer un changement par la modification de ces paramètres (si non on accède à la dernière copie par défaut)
#########################################################################################################
root = './data'
sampling_BS = 1000
k_kouches = 3
num_neighbors = [-1]*k_kouches
num_samples = 30000
n_chunks = 20
DS = MyOwnDataset(root, sampling_BS, num_neighbors, num_samples, n_chunks)
#########################################################################################################
## parametres spécifiques à l'entrainement du modèle DeepGraphInfomax
#########################################################################################################
n_epoch = 50
in_channels = 969
hidden_channels = 512
channels = [in_channels]+([hidden_channels]*k_kouches)
train_BS = 512
#########################################################################################################


processed_dir = [os.path.join(root,"processed",u) for u in os.listdir(os.path.join(root,"processed"))]
train_list = []
#for idx in range(DS.length()):
for idx in range(3):
    train_list.extend(get(processed_dir,idx))

train_loader = DataLoader(train_list,batch_size = train_BS,shuffle=True)

print(train_loader.batch_size)

model = DeepGraphInfomax(
    hidden_channels = channels[-1],
    encoder=Encoder(channels),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption
                        )
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



for epoch in range(n_epoch):
    print('epoch : {:d}'.format(epoch+1))
    loss = train(model,optimizer,epoch,train_loader)
    print('Loss:', loss)



# On test avec train_loader car non supervisé 
train_ratio = 0.75
n_estim = 1000
A1,A2,A3 = test(model,train_loader,sampling_BS,train_ratio,n_estim)
print('Test Accuracy (features ,DGI only , DGI+features):' A1,A2,A3)
