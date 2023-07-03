import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import csv

from model import Resnet18
from dataset import TestDataset
from metric import supervisedContrasLoss,KNN




def finetune(model,num_epochs,savepath):
    model.train()
    
    for task,(sup_image ,sup_label, qry_image)in enumerate(testloader):
        model.load_state_dict(torch.load(savepath))
        acc = []
        for epoch in (bar:=tqdm(range(num_epochs),ncols=0)):
        
            sup_images = sup_image[0].to(device)  # 20x3x84x84

            sup_images1 = transform(sup_images)
            sup_images2 = transform(sup_images)
            sup_images_cat = torch.cat((sup_images1, sup_images2), 0)

            sup_labels = sup_label[0].to(device)  # 20
            sup_labels_cat = torch.cat((sup_labels, sup_labels), 0)

            optimizer.zero_grad()

            output = model(sup_images_cat)

            loss = supervisedContrasLoss(output, sup_labels_cat)

            loss.backward()
            optimizer.step()

            accuracy = KNN(output, sup_labels_cat,  Ks=[8, 10])
            acc.append(accuracy)
            
            bar.set_description(f'tasks[{task+1}/600]|epoch[{epoch+1:3d}/{num_epochs}]|Finetuning')
            bar.set_postfix_str(
                f' loss {loss.item():.4f} knn acc {sum(acc)/len(acc) :.4f} ')
            
            
        pred=distance(model,qry_image,sup_image ,sup_label)
        preds.extend(pred)
            
            
            
        



@torch.no_grad()
def distance(model,qry_image,sup_images, sup_labels):
    model.eval()
    cla_list = []

    sup_images = sup_images[0].to(device)  # 25x3x84x84
    sup_labels = sup_labels[0].to(device)  # 25
    qry_image = qry_image[0].to(device)  # 25x3x84x84

    sup_output = model(sup_images)  # 25xdim
    cla_mean = np.mean([sup_output.cpu().numpy()[
                    (sup_labels == c).cpu().numpy()] for c in range(5)], axis=1)  # 5xdim

    qry_output = model(qry_image).cpu().numpy()  # 25xdim

    dis = np.sum((qry_output[None, :, :]-cla_mean[:, None, :])**2, axis=-1)

    cla = np.argmin(dis, axis=0)  # 25
    #sim=qry_output@cla_mean.T#25X5
    
    #cla=np.argmax(sim,axis=-1)
    cla_list.extend(cla)
    return cla_list




if __name__=='__main__':
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(30, [0.2, 0.2], [
                            0.8, 1.2], ),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),

    ])
    testdata = TestDataset()
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)

    model = Resnet18()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    savepath = "best.pth"
    preds=[]

    num_epochs=50
    
    finetune(model,num_epochs,savepath)

    data = [(i, val) for i, val in enumerate(preds)]


    with open('311513058_pred.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        writer.writerows(data)