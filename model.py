

import torch
from torch import nn
from torch_geometric.nn import  SAGEConv
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import torch.nn.functional as F
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier

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


@torch.no_grad()
def test(model,test_loader,sampling_BS,train_ratio,n_estim):
    model.eval()
    Ids = []
    Z = []
    Y = []
    X = []
    for data in tqdm(test_loader.dataset):
        Ids.extend([(data.edge_original[0][i].item(),data.edge_original[1][i].item()) for i in range(sampling_BS)])
        Zs.append(model(data.x, data.edge_index)[:sampling_BS])
        Xs.append(data.x[:sampling_BS])
        Ys.extend(data.y[:sampling_BS].tolist())

    X = torch.cat(Xs, dim=0)
    Z = torch.cat(Zs, dim=0)
    Y = torch.tensor(Ys)

    train_idx = random.choices([i for i in range(len(Y)],int(len(Y)*train_ratio))
    test_idx = list(set([i for i in range(len(Y))])-set(train_idx))

    x_train,x_test = X[train_idx],X[test_idx]
    z_train,z_test = Z[train_idx],Z[test_idx]
    y_train,y_test = Y[train_idx],Y[test_idx]

    sm = SMOTE(sampling_strategy="auto")

    x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)
    classifier_1 = AdaBoostClassifier(n_estimators=n_estim).fit(x_train_sm, y_train_sm)
    acc_1 = classifier_1.score(x_test,y_test)

    z_train_sm, y_train_sm = sm.fit_resample(z_train, y_train)
    classifier_2 = AdaBoostClassifier(n_estimators=n_estim).fit(z_train_sm, y_train_sm)
    acc_2 = classifier_2.score(z_test,y_test)


    xz_train_sm, y_train_sm = sm.fit_resample(torch.cat((x_train,z_train),dim=1), y_train)
    classifier_3 = AdaBoostClassifier(n_estimators=n_estim).fit(xz_train_sm, y_train_sm)
    acc_3 = classifier_3.score(torch.cat((x_test,z_test),dim=1),y_test)


    return acc_1, acc_2, acc_3
