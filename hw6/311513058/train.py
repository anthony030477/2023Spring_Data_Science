import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import csv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(10, 32,)
        self.conv2 = GCNConv(32, 64,)
        self.fc = nn.Linear(74, 1)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.conv2(x1, edge_index)
        x1 = F.leaky_relu(x1)
        x = torch.cat([x, x1], dim = -1)
        
        x = self.fc(x)
        
        return torch.sigmoid(x.view(-1))



def train(model, data, labels, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-5)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    # loss_list=[]
    data=data.to(device)
    labels=labels.to(device)
    mvloss=10
    best_auc=0.83
    for epoch in (bar:=tqdm(range(epochs),ncols=0)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[train_mask]
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        # loss_list.append(loss.item())
        acc=((out>0.5)==(labels>0.5)).float().mean().item()
        f1=f1_score(labels.cpu().detach().numpy(),(out>0.5).int().cpu().detach().numpy())
        auc=roc_auc_score(labels.cpu().detach().numpy(),out.cpu().detach().numpy())
        if auc>best_auc:
            torch.save(model.state_dict(), 'save.pt')
            best_auc=auc
        mvloss=0.95*mvloss+0.05*loss.item()
        bar.set_description(f'epoch[{epoch+1:3d}/{epochs}]|Training')
        bar.set_postfix_str(f'loss: {mvloss:.4f}, acc{acc:.3f},f1: {f1:.3f},auc: {auc:.3f}')
        scheduler.step()
    print(f'best_auc:{best_auc}')


train_data=torch.load('dataset/train_sub-graph_tensor.pt')#Data(edge_index=[2, 6784824], feature=[39357, 10], label=[15742])
train_mask=np.load('dataset/train_mask.npy')
train_mask=torch.from_numpy(train_mask)

edge_index = train_data['edge_index'].to(torch.long)

features = train_data['feature'].to(torch.float32)

labels =train_data['label'].to(torch.float)


data = Data(x=features, edge_index=edge_index)

model = GCN()
model.to(device)
train(model, data, labels, epochs=10000)



#eval
test_data=torch.load('dataset/test_sub-graph_tensor_noLabel.pt')#Data(edge_index=[2, 7000540], feature=[39357, 10])

test_mask=np.load('dataset/test_mask.npy')

test_mask=torch.from_numpy(test_mask)


edge_index = test_data['edge_index'].to(torch.long)
features = test_data['feature'].to(torch.float32)

data = Data(x=features, edge_index=edge_index)

model = GCN()
model.to(device)
model.load_state_dict(torch.load('save.pt'))

def eval(model, data):
    model.eval()
    data=data.to(device)
    out = model(data.x, data.edge_index)
    # out=F.sigmoid(out)
    return out[test_mask]

submission=eval(model, data)
index=torch.arange(len(test_mask))[test_mask]

data = [(i.item(), val.item()) for i, val in zip(index,submission)]


with open('311513058_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['node idx', 'node anomaly score'])  
    writer.writerows(data)  


