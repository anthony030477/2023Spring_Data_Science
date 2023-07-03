import torch
from tqdm import tqdm
import torchvision
from dataset import trainDataset
from model import Unet,VGG,Wnet
from torchvision.utils import save_image
import torchvision.transforms as transforms

config={
    'lr':2e-5,
    'epochs':2000,
    'min_lr':8e-6,
}

def train(i,num_epochs,model):
    model.train()
    loss_list=[]
    label_loss=[]
    for images,points,st_sizes,labels in( bar :=tqdm(trainloader,ncols=0)):
        
        images = images.to(device)
        # heat_map=heat_map.to(device)
        points=[point.to(device) for point in points]
        # save_image(heat_map[0], 'heat_map.png')
        
        labels = labels.to(device)#batchsize 
        
        optimizer.zero_grad()
        
        output_map=model(images)
        prob_list=posterior_prob(points, st_sizes,device)
        bay_loss=bayesian_loss(prob_list,[torch.ones(label.int(),device=device) for label in labels],output_map,device)
        #bay_loss(prob_list, target_list=[torch.ones(label.int(),device=device) for label in labels], pre_density=output_map)
        # save_image(output_map[0], 'output_map.png')
        with torch.no_grad():
            predict_label=torch.sum(output_map,dim=(1, 2, 3))
            loss_mae=criterion_mae(predict_label, labels)
        
        #loss=0.1*loss_mae+bay_loss
        
        # loss=config['mse_loss']*criterion_mse(output_map, heat_map)+\
        #     config['bce_loss']*criterion_bce(x_branch,heat_map)+\
        #        config['mae_loss']*loss_mae
        # loss =criterion_mse(output_map, heat_map)+ config['mae_loss']*loss_mae
        # mask=heat_map>config['threshold']
        # loss=criterion1(output_map, heat_map)
        # loss=torch.mean(config['positive']*torch.sum(loss*mask,dim=(1,2,3))/(torch.sum(mask,dim=(1,2,3))+10)+\
        #                 config['nagative']*torch.sum(loss*(~mask),dim=(1,2,3))/torch.sum(~mask,dim=(1,2,3)))+\
        #                                                 config['mae_loss']*a
        bay_loss.backward()
       
        optimizer.step()
       

        loss_list.append(bay_loss.item())
        label_loss.append(loss_mae.item())
        bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
        bar.set_postfix_str(f'loss {sum(loss_list)/len(loss_list):.4f} mae score{sum(label_loss)/len(label_loss):.2f}predict {int(predict_label[0]):3d} target{int(labels[0]):3d}')
    scheduler.step()


def val(model):
    model.eval()
    
    label_loss=[]
    with torch.no_grad():
        for images,labels in( bar :=tqdm(valoader,ncols=0)):
            
            images = images.to(device)
            # heat_map=heat_map.to(device)
            labels = labels.to(device)

            output_map=model(images)
            predict_label=torch.sum(output_map,dim=(1,2,3))
            # predict_label=0
            # for img in images:
            #     img=img.unsqueeze(0).to(device)
            #     output_map=model(img)
            #     predict_label+=torch.sum(output_map)
            loss_mae=criterion_mae(predict_label, labels)
            
            
            label_loss.append(loss_mae.item())
            bar.set_description(f'validation')
            bar.set_postfix_str(f'mae score {sum(label_loss)/len(label_loss):.4f} predict {int(predict_label[0]):3d} target{int(labels[0]):3d}')
    return sum(label_loss)/len(label_loss)


def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1] 
    st_sizes = torch.FloatTensor(transposed_batch[2])
    label= torch.FloatTensor(transposed_batch[3])
    return images, points,  st_sizes,label
def bayesian_loss(post_prob,targets,output_map,device):
    loss = 0
    for idx, prob in enumerate(post_prob):  
        if prob != None:
            N = len(prob)
            target = torch.cat([targets[idx].to(device),
                    torch.zeros(N - len(targets[idx])).to(device)]).to(device).float()
            count = torch.sum(output_map[idx].view((1, -1)) * prob, dim=1)  
        else:  
            count = torch.sum(output_map[idx])
            target = torch.tensor([0.]).to(device).float()
        criterion = torch.nn.L1Loss(reduction='none')
        loss += torch.sum(criterion(count, target))
    loss = loss / len(post_prob)
    return loss

def posterior_prob(points, st_sizes,device):
    coord = torch.linspace(0, 512-8, 64).float()+ 8 / 2
    coord=coord.unsqueeze(0).to(device).float()
    softmax = torch.nn.Softmax(dim=0)
    prob_list = []
    num_points_per_image =list(map(len, points))
    all_points = torch.cat(points, dim=0)

    if len(all_points) > 0:
        x = all_points[:, 0].unsqueeze(1)
        y = all_points[:, 1].unsqueeze(1)

        x_dis = -2 * x@coord + x * x + coord * coord
        y_dis = -2 * y@coord+ y * y + coord * coord

        dis = y_dis.unsqueeze(2) + x_dis.unsqueeze(1)
        dis = dis.view((dis.size(0), -1))

        dis_list = torch.split(dis, num_points_per_image)
       
        for dis, st_size in zip(dis_list, st_sizes):
            if len(dis) > 0:
                min_dis = torch.min(dis, dim=0, keepdim=True)[0]
                min_dis = torch.where(min_dis < 0, torch.zeros_like(min_dis)+1e-5, min_dis+1e-5)
                dis = torch.cat([dis, ((st_size*1) ** 2) / min_dis], 0) 
                prob = softmax(-dis / (2.0 * 8 ** 2))
            else:
                prob = None
            prob_list.append(prob)
    else:
        prob_list = [None] * len(points)
        
    return prob_list

traindata=trainDataset()
trainloader=torch.utils.data.DataLoader(traindata, batch_size=4,shuffle=True,num_workers=12,collate_fn=collate_fn)

valdata=trainDataset(train=False)
valoader=torch.utils.data.DataLoader(valdata, batch_size=4,shuffle=True,num_workers=12)

model=VGG()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
# model.load_state_dict(torch.load('save.pth'))

optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=1e-4,)


# criterion_bce = torch.nn.BCELoss()
# criterion_mse=torch.nn.MSELoss()
criterion_mae = torch.nn.L1Loss()
# criterion_smool1=torch.nn.SmoothL1Loss()
#criterion2=torch.nn.CrossEntropyLoss()

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=8e-6,patience=20)
scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['min_lr'])





num_epochs=config['epochs']
best_mae=1000
for i in range(num_epochs):
    train(i,num_epochs,model)
    
    mae_score=val(model)
    if mae_score<best_mae:
        torch.save(model.state_dict(), 'save.pth')
        best_mae=mae_score

print(f'best mae: {best_mae}')