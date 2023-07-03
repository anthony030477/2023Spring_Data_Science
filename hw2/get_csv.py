import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision.models import resnet50
from tqdm import tqdm
import csv
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
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


savepath='save\save_para_0.9448.pth'
    
    
student.load_state_dict(torch.load(savepath))
student.to(device)

student.eval()
acc=[]
output_list=[]
with torch.no_grad():
    for images, _ in( bar :=tqdm(testloader,smoothing=0.1)):
        
        images = images.to(device)
        
        student_outputs=student(images)
        output=torch.argmax(student_outputs[2], dim=1).cpu().numpy()
        
        
        output_list.extend(output)

data = [(i, val) for i, val in enumerate(output_list)]


with open('311513058_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'pred'])  
    writer.writerows(data)  