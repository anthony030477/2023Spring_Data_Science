import torch

import torch.nn.functional as F



def supervisedContrasLoss(pred, label):
    '''
    pred:N,D
    label:N, 
    '''
    pred = pred@pred.T  # N,N
    pred = F.softmax(pred/5,dim=1)
    mask = label[None, :] == label[:, None]  # N,N
    sele = torch.masked_select(pred, mask)
    return -torch.mean(torch.log(sele+1e-12))

def KNN(emb,label,Ks=[1,5,10]):
    '''
    emb:N,dim
    label:N,
    '''
    dist=torch.sum((emb[None,:,:]-emb[:,None,:])**2,dim=-1)#N,N
    mask=torch.eye(emb.size(0)).to(emb.device)#N,N
    mask=mask==1
    dist = torch.masked_fill(dist, mask, float('inf'))#N,N
    preds=[]
    for K in Ks:
        _,k_indices = dist.topk(K, dim=1, largest=False)#N,K
        knn = label[k_indices]#N,K
        knn_labels_val,ind=torch.mode(knn)#N
        preds.append(knn_labels_val)
    accs=[]
    for pred in preds:
        acc=(pred==label).sum().item()/emb.size(0)
        accs.append(acc)
    return max(accs)

