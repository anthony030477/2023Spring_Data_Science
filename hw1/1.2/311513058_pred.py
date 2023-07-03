import torch
from torchvision import transforms
import cv2
from tqdm import tqdm
import click
from torchvision.models import resnet18, ResNet18_Weights

@click.command()
@click.argument('image_path_list')
def pred(image_path_list):
    class Beautyregression(torch.nn.Module):
        def __init__(self):
            super(Beautyregression,self).__init__()
            
            #self.sigmoid = torch.nn.Sigmoid()
            self.resnet18 = resnet18(weights = ResNet18_Weights.DEFAULT, progress = False)
            self.resnet18.fc = torch.nn.Linear(512, 1)
        def forward(self,x):
            # return x
            x = self.resnet18(x)
            # print(x.size())
            #x = self.sigmoid(x)
            return x



    model_beauty = Beautyregression()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_beauty.to(device)


    model_beauty.load_state_dict(torch.load('best_reg_0.9327251995438997.pth'))


    class TestDataset:
        def __init__(self, file_path=image_path_list):
        
            with open(file_path, 'r') as f:
                lines = f.readlines()

            self.image_paths = []
                    
            for line in lines:
                path = line.strip()
                self.image_paths.append(path)
        
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
        
            image_path = self.image_paths[index]
            #image = transforms.ToTensor()(Image.open(image_path))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#hwc
            image = cv2.resize(image, (224, 224))
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)  # chw
            image = image.float() 
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
            
            return image, 0


    testdataset=TestDataset()
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=10, shuffle=False)

    model_beauty.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            outputs = model_beauty(images)
           # print(outputs)
            threshold_tensor = torch.tensor(35.0).cuda()

            binary_outputs = torch.where(outputs.squeeze(1) > threshold_tensor, torch.ones_like(outputs.squeeze(1)), torch.zeros_like(outputs.squeeze(1)))
            #print(binary_outputs)
            predictions.extend(binary_outputs.cpu().numpy().astype(int))
        


    
    with open('311513058.txt', 'w') as f:
        f.write(''.join(str(p) for p in predictions))



if __name__ == '__main__':
    pred()
