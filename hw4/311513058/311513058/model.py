import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
       
        
      
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            

            nn.Conv2d(64, 64, 3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            

            nn.Conv2d(256, 256, 3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
           

            nn.Conv2d(512, 512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.back_end=nn.Sequential(
                nn.Conv2d(512,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,128,3,1,1),
                # nn.PixelShuffle(2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,1,1),
                # nn.PixelShuffle(2),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(64,32,3,1,1),
                # # nn.PixelShuffle(2),
                # nn.BatchNorm2d(32),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(32,1,3,1,1),
                # # nn.PixelShuffle(2),
                
        )
        # self.relu=nn.ReLU()
    def forward(self,x):
        
        x=self.features(x)
        x = F.interpolate(x, scale_factor=2)
        x=self.back_end(x)
        x = torch.abs(x)

       
         
        
        return x
    



     

class Unet(nn.Module):
        def __init__(self):
            super(Unet, self).__init__()
            self.upcov1=nn.Sequential(
                 nn.Conv2d(3,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                
                 nn.Conv2d(64,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),

                 nn.Conv2d(64,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                 
            )
            self.upcov2=nn.Sequential(
                 nn.Conv2d(64,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
            
                 nn.Conv2d(128,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),

                 nn.Conv2d(64,32,3,1,1),
                 nn.BatchNorm2d(32),
                 nn.ReLU(),

                 nn.Conv2d(32,1,3,1,1),
                #  nn.BatchNorm2d(1),
                 
            )
            self.midcov1=nn.Sequential(
                 nn.Conv2d(64,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                 

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                 
            )
            self.midcov2=nn.Sequential(
                 nn.Conv2d(128,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),

                 nn.Conv2d(256,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                #  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                 nn.Conv2d(128,64*4,3,1,1),
                 nn.PixelShuffle(2)
            )
            self.downcov1=nn.Sequential(
                 nn.Conv2d(128,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),

                 nn.Conv2d(256,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),
                 
                 nn.Conv2d(256,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),
                #  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                 nn.Conv2d(256,128*4,3,1,1),
                nn.PixelShuffle(2)
            )
           
            self.maxpool=nn.MaxPool2d(2,2)
            self.cnn1=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
        )
            self.relu=nn.ReLU()
            self.fc = nn.Linear(524288, 1)
            # self.dropout = nn.Dropout(p=0.5)
            self.adpative=nn.AdaptiveMaxPool2d((1,1))
        def forward(self, x):
            x=self.upcov1(x)#bx64x512x512
            copy1=x
            x=self.maxpool(x)#bx64x256x256

            x=self.midcov1(x)#bx128x256x256
            copy2=x
            x=self.maxpool(x)#bx64x128x128
            
            x=self.downcov1(x)#bx128x256x256
            

            x=x+copy2#bx128x256x256
            x=self.midcov2(x)#bx64x512x512
            x=x+copy1
            x=self.upcov2(x)
            heat_map=x
            # label= self.adpative(self.relu(heat_map))
            # x=self.cnn1(x)
            
            # x = x.view(x.size(0), -1)
            # # x=self.dropout(x)
            # x=self.fc(x)
            return self.relu(heat_map)


class Wnet(nn.Module):
        def __init__(self):
            super(Wnet, self).__init__()
            self.upcov1=nn.Sequential(
                 nn.Conv2d(3,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                
                 nn.Conv2d(64,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),

                 nn.Conv2d(64,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),
                 
            )
            self.upcov2=nn.Sequential(
                 nn.Conv2d(64,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
            
                 nn.Conv2d(128,64,3,1,1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(),

                 nn.Conv2d(64,32,3,1,1),
                 nn.BatchNorm2d(32),
                 nn.ReLU(),

                 nn.Conv2d(32,1,3,1,1),
                #  nn.BatchNorm2d(1),
                 
            )
            self.midcov1=nn.Sequential(
                 nn.Conv2d(64,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                 

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                 
            )
            self.midcov2=nn.Sequential(
                 nn.Conv2d(128,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),

                 nn.Conv2d(256,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),

                 nn.Conv2d(128,128,3,1,1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(),
                #  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                 nn.Conv2d(128,64*4,3,1,1),
                 nn.PixelShuffle(2)
            )
            self.downcov1=nn.Sequential(
                 nn.Conv2d(128,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),

                 nn.Conv2d(256,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),
                 
                 nn.Conv2d(256,256,3,1,1),
                 nn.BatchNorm2d(256),
                 nn.ReLU(),
                #  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                 nn.Conv2d(256,128*4,3,1,1),
                nn.PixelShuffle(2)
            )
           
            self.maxpool=nn.MaxPool2d(2,2)
            self.cnn1=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
        )
            self.dropout=nn.Dropout2d(p=0.2)
            self.relu=nn.ReLU()
            self.fc = nn.Linear(524288, 1)
            # self.dropout = nn.Dropout(p=0.5)
            self.adpative=nn.AdaptiveMaxPool2d((1,1))
            self.convbranch=nn.Conv2d(1,1,3,1,1)
        def forward(self, x):
            x=self.upcov1(x)#bx64x512x512
            copy1=x
            x=self.maxpool(x)#bx64x256x256

            x=self.midcov1(x)#bx128x256x256
            copy2=x
            x=self.maxpool(x)#bx64x128x128
            
            x=self.downcov1(x)#bx128x256x256
            
            x_branch=x
            x_branch=x_branch+copy2#bx128x256x256
            x_branch=self.midcov2(x_branch)#bx64x512x512
            x_branch=x_branch+copy1
            x_branch=self.upcov2(x_branch)
            x_branch=self.convbranch(x_branch)
            x_branch=torch.sigmoid(x_branch)

            x=x+copy2#bx128x256x256
            x=self.midcov2(x)#bx64x512x512
            x=x+copy1
            x=self.upcov2(x)
            x=self.relu(x)
            heat_map=x*x_branch
            
            return heat_map,x_branch
if __name__=='__main__':
    model=VGG()
    x=torch.randn((1,3,512,512))
    y=model(x)
    
    # print(model)
    print(y.shape)
    # print(model.vgg.state_dict().keys())