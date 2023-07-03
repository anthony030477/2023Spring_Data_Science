import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision.models import resnet50
from tqdm import tqdm
if __name__=='__main__':
    transform = transforms.Compose([
        transforms.RandAugment(2,5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform1 = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                shuffle=True, num_workers=6)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=0)
  

    

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.cnn1=nn.Sequential(
                nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(28),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            self.cnn2=nn.Sequential(
                nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(56),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            self.cnn3=nn.Sequential(
                nn.Conv2d(56, 136, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(136),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                
            )
            self.classifier0 = nn.Sequential(
                
                nn.Conv2d(28, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((28, 28))#256x7x7
            )
            self.classifier1 = nn.Sequential(
                
                nn.Conv2d(136, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((16,16 ))#1024x2x2
            )

            
            self.fc = nn.Linear(136*3*3, 10)
            self.dropout = nn.Dropout(p=0.1)
            
        def forward(self, x):
            x=self.cnn1(x)
            x0=self.classifier0(x)
            x0=x0.view(x0.size(0),-1,7,7)

            x=self.cnn2(x)

            x=self.cnn3(x)
            x1=self.classifier1(x)
            x1=x1.view(x1.size(0),-1,2,2)

            x = x.view(x.size(0), -1)
            x=self.dropout(x)
            x=self.fc(x)
            
            return x0,x1,x

    student= CNN()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Teachermodel(torch.nn.Module):
        def __init__(self):
            super(Teachermodel,self).__init__()
        
            self.resnet50 =resnet50(pretrained=False)
            self.resnet50.fc = torch.nn.Linear(2048, 10)
        def forward(self,x):
            x = self.resnet50.conv1(x)
            x = self.resnet50.bn1(x)
            x = self.resnet50.relu(x)
            x = self.resnet50.maxpool(x)

            x = self.resnet50.layer1(x)
            x0=x#256x7x7
            x = self.resnet50.layer2(x)
            x = self.resnet50.layer3(x)
            x1=x#1024x2x2
            x = self.resnet50.layer4(x)
            x = self.resnet50.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.resnet50.fc(x)
            x=F.softmax(x/2.5,dim=1)
            return x0,x1,x

    teacher_dict = torch.load('model-compression-on-fashion-mnist\\resnet-50.pth')
    teacher=Teachermodel()
    #print(teacher_dict.keys())
    teacher.load_state_dict(teacher_dict['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher.to(device)

    student.to(device)
    optimizer = torch.optim.Adam(student.parameters(),lr=0.001)
    
    criterion = torch.nn.CrossEntropyLoss()
    criterion2=torch.nn.MSELoss()
    summary(student, input_size=(3,28 , 28))

    def train(i):
        student.train()
        teacher.eval()
        acc=[]
        for epoch in range(1):
            for images, labels in( bar :=tqdm(trainloader)):
                
                images = images.to(device)
                labels = labels.to(device)#batchsize 
                
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                student_outputs=student(images)
                # teacher_acc=(( torch.argmax(teacher_outputs[2], dim=1)==labels).sum().item())/labels.size(0)
                # if teacher_acc>0.95:

                #     loss = 0.01*criterion(student_outputs[0], teacher_outputs[0])+0.02*criterion(student_outputs[1], teacher_outputs[1])+0.97*criterion(student_outputs[2], teacher_outputs[2])
                # else:
                #     target = torch.argmax(torch.eye(labels.shape[0]).to(device)[labels], dim=1)
                #     loss = criterion(student_outputs[2], target)
                target = torch.argmax(torch.eye(labels.shape[0]).to(device)[labels], dim=1)
                loss=0.49*criterion(student_outputs[2], target)+0.01*criterion2(student_outputs[0], teacher_outputs[0])+0.01*criterion2(student_outputs[1], teacher_outputs[1])+0.49*criterion(student_outputs[2], teacher_outputs[2])
                # print(labels.size())
                # input('e')
                loss.backward()
                optimizer.step()

        
                accuracy=(( torch.argmax(student_outputs[2], dim=1)==labels).sum().item())/labels.size(0)
                acc.append(accuracy)

                bar.set_description(f'epoch[{i+1:3d}/{num_epochs}]|Training')
                bar.set_postfix_str(f' loss {loss.item():.4f} accuracy {sum(acc)/len(acc) :.4f}')

    def test():
        student.eval()
        acc=[]
        total=[]
        with torch.no_grad():
            
            for images, labels in( bar :=tqdm(testloader,smoothing=0.1)):
                
                images = images.to(device)
                labels = labels.to(device)#batchsize 
                
                student_outputs=student(images)
                
                
                accuracy=(( torch.argmax(student_outputs[2], dim=1)==labels).sum().item())#/labels.size(0)

                acc.append(accuracy)
                total.append(labels.size(0))
            
            print("accuracy: ",sum(acc)/sum(total))
            return sum(acc)/sum(total)

    num_epochs=1000
    accc=0.94
    for i in range(num_epochs):
        train(i)

        accu= test()
        if accu>=accc:
            torch.save(student.state_dict(), 'save/save_para_'+str(accu)+'.pth')
            accc=accu


    