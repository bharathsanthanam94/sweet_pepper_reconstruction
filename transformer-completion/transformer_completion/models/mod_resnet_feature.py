"""
A small script to extract features from Resnet50 architecture
Input: Image (Pre-processing not performed)
Output: Features from the specified layer

"""
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import warnings
import ipdb
import time
warnings.filterwarnings('ignore')

def extract_features(model_type,input_img):
    #use pretrained resnet on Imagenet
    resnet = models.resnet50(pretrained=True)

    #Uncomment to use pretrained resnet through SSL (DINO model)
    # resnet50= torch.hub.load('facebookresearch/dino:main','dino_resnet50')

    device=torch.device("cuda")
    resnet=resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    layers = list(resnet.children())[:6]  # up to "layer2"
    partial_model = nn.Sequential(*layers).to(device)

    img_tensor = transform(input_img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        layer_features = partial_model(img_tensor)
        
    return layer_features

if __name__ == "__main__":
    image_path="/home/bharath/Desktop/thesis/code/pytorch3D_proj/00001.png"
    image = Image.open(image_path)
    features = extract_features("resnet50", image)
    