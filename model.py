

import torch
import numpy as np
from torch import nn
from torch_geometric.nn import  SAGEConv
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import torch.nn.functional as F
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb 
import random 
from sklearn.utils.class_weight import compute_sample_weight

class Encoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        # Channels contient les dimensions de toutes les couches en incluant la dimension d'entrée ( = channels[0])
        self.convs = torch.nn.ModuleList([SAGEConv(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.activations = torch.nn.ModuleList([nn.PReLU(channels[i]) for i in range(1,len(channels))])


    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x = self.convs[i](x.float(),edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def summary(z,*args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))

def train(model,optimizer,epoch,train_loader):
    model.train()

    total_loss = total_examples = 0
    for data in tqdm(train_loader.dataset):

        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)

    return total_loss / total_examples


class XGboost_classifier():
    def __init__(self,Graph_representation_model,test_loader,train_ratio,max_depth,tree_method,learning_rate,n_estimators):
        self.classifier_model = xgb.XGBClassifier(max_depth = max_depth,tree_method = tree_method,learning_rate = learning_rate,n_estimators = n_estimators )
        self.Graph_representation_model = Graph_representation_model
        self.test_loader = test_loader
        self.train_ratio = train_ratio
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def embed(self):
        print("using model to embed : ")
        self.Graph_representation_model.eval()
        Ids = []
        Zs = []
        Ys = []
        Xs = []
        for data in tqdm(self.test_loader.dataset):
            Ids.extend([(data.edge_original[0][i].item(),data.edge_original[1][i].item()) for i in range(self.Graph_representation_model.sample_bs)])
            pos_z,neg_z,summary = self.Graph_representation_model(data.x, data.edge_index)
            Zs.append(pos_z[:self.Graph_representation_model.sample_bs])
            Xs.append(data.x[:self.Graph_representation_model.sample_bs])
            Ys.extend(data.y[:self.Graph_representation_model.sample_bs].tolist())

        X = torch.cat(Xs, dim=0).detach().numpy()
        Z = torch.cat(Zs, dim=0).detach().numpy()
        Y = torch.tensor(Ys).detach().numpy()

        return X,Y,Z
    
    def define_idx(self,Y):
        train_idx = random.choices([i for i in range(len(Y))], k = int(len(Y)*self.train_ratio))
        test_idx = list(set([i for i in range(len(Y))])-set(train_idx))
        return train_idx,test_idx

    def train_classifiers(self,X,Y,Z):

        train_idx,test_idx = self.define_idx(Y)
        x_train,x_test = X[train_idx],X[test_idx]
        z_train,z_test = Z[train_idx],Z[test_idx]
        y_train,y_test = Y[train_idx],Y[test_idx]

        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train )

        classifier_model = self.classifier_model
    
        print("training classifier based on features only : ")
        classifier_1 = classifier_model.fit(x_train, y_train,sample_weight=sample_weights)
        predicted_1 = classifier_1.predict(x_test)
        recall_1, precision_1 = recall_score(y_test,predicted_1), precision_score(y_test,predicted_1)
    
        print("training classifier based on embeddings only : ")
        classifier_2 = classifier_model.fit(z_train, y_train,sample_weight=sample_weights)
        predicted_2 = classifier_2.predict(z_test)
        recall_2, precision_2 = recall_score(y_test,predicted_2), precision_score(y_test,predicted_2)

        print("training classifier based on embeddings + fetaures : ")
        classifier_3 =  classifier_model.fit(np.concatenate((x_train,z_train),axis=1), y_train,sample_weight=sample_weights) 
        predicted_3 = classifier_3.predict(np.concatenate((x_test,z_test),axis=1))
        recall_3, precision_3 = recall_score(y_test,predicted_3), precision_score(y_test,predicted_3)

        return recall_1, precision_1, recall_2, precision_2, recall_3, precision_3
