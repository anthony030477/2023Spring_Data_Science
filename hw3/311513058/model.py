import torch
from torchvision.models import resnet18


class Resnet18(torch.nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        self.resnet18 = resnet18(weights=None)
        #self.resnet50.fc = torch.nn.Linear(512, 32)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)

        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)

        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        # feature=x
        x = self.resnet18.fc(x)
        # feature=x
        #x = torch.sigmoid(x)
        feature = x
        return feature