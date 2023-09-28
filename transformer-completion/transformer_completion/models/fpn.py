'''
Script that builds FPN on top of ResNet architecture
Adapted from : https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ResNetFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone= resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=5)

    
    def forward(self,x):
        output=self.backbone(x)
        layer4,layer3= output['0'],output['1']
        #start with just layer3, which is of shape [1,256,90,160]
        return layer3


#test the above model
# get some dummy image
if __name__ =="__main__":
    #assign some dummy image tensor
    x = torch.rand(1,3,720,1280)
    resnetfpn_features=ResNetFPN()
    layer3= resnetfpn_features(x)
    import ipdb;ipdb.set_trace()






'''
#how to access outputs?
print([(k, v.shape) for k, v in output.items()])
        # returns
         [('0', torch.Size([1, 256, 16, 16])),
           ('1', torch.Size([1, 256, 8, 8])),
           ('2', torch.Size([1, 256, 4, 4])),
           ('3', torch.Size([1, 256, 2, 2])),
           ('pool', torch.Size([1, 256, 1, 1]))]

'''
