import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm


from model import Resnet18
from dataset import MyDataset
from metric import KNN,supervisedContrasLoss


transform = transforms.Compose([

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(30, [0.2, 0.2], [
                            0.8, 1.2], shear=(0, 0, 0, 45)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    #transforms.ColorJitter(0.3,0.3,0.3,0.3),
])

    
traindata = MyDataset()
trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=256, shuffle=True)

model = Resnet18()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(i, num_epochs, model):
    model.train()
    acc = []

    for epoch in range(1):
        for images, labels in (bar := tqdm(trainloader,ncols=0)):
            images = torch.cat((images, images), 0)
            # images=transform(images)

            images = images.to(device)
            images = transform(images)
            labels = labels.to(device)  # batchsize

            labels = torch.cat((labels, labels), 0)
            optimizer.zero_grad()

            output = model(images)

            loss = supervisedContrasLoss(output, labels)
            loss.backward()
            optimizer.step()

            accuracy = KNN(output, labels, Ks=[8])
            acc.append(accuracy)

            bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
            bar.set_postfix_str(
                f' loss {loss.item():.4f} accuracy {sum(acc)/len(acc) :.4f}')
    return sum(acc)/len(acc)

num_epochs = 500
accuu = 0.05
for i in range(num_epochs):
    accu = train(i, num_epochs, model)
    #print('acc:', accu)
    if accu > accuu:
        torch.save(model.state_dict(), 'best.pth')
        accuu = accu
print('best acc :',accuu)