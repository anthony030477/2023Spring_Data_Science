import csv
import torch
from tqdm import tqdm

from dataset import testDataset
from model import Unet,Wnet,VGG

def test(model):
    model.eval()
    
    out=[]
    with torch.no_grad():
        
        for images, _ in( bar :=tqdm(testloader,smoothing=0.1)):
            images = images.to(device)
            output=model(images)

            predict=torch.sum(output,dim=(1, 2, 3))
            a=predict.cpu().numpy()
            out.extend(a)

    return out






testdata=testDataset()
testloader=torch.utils.data.DataLoader(testdata, batch_size=1,shuffle=False)


model=VGG()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.load_state_dict(torch.load('save.pth'))

output_list=test(model)

data = [(i+1, val) for i, val in enumerate(output_list)]


with open('311513058_pred.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Count'])  
    writer.writerows(data)  