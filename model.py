import torch
from torch import nn
from torch.nn import functional as F

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,i_downsample=None, stride=1):
        super(Bottleneck,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels*4,kernel_size=1,stride=1,padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self,x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        if self.i_downsample is not None:
            identity =  self.i_downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self,res_block, layers, num_classes, num_channels=3):
        super(ResNet,self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(res_block=res_block,num_blocks=layers[0],planes=64)
        self.layer2 = self._make_layer(res_block=res_block,num_blocks=layers[1],planes=128,stride=2)
        self.layer3 = self._make_layer(res_block=res_block,num_blocks=layers[2],planes=256,stride=2)
        self.layer4 = self._make_layer(res_block=res_block,num_blocks=layers[3],planes=512,stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        
        return x
    
    
    def _make_layer(self,res_block, num_blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*4:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*4)
            )
        
        layers.append(res_block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * 4
        
        for i in range(num_blocks-1):
            layers.append(res_block(self.in_channels,planes))
        
        return nn.Sequential(*layers)
        
        
        
def ResNet50(num_classes,channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_channels=channels, num_classes=num_classes)

