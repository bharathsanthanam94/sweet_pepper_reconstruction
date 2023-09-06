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
warnings.filterwarnings('ignore')
def extract_features(model_type, input_img):
    if model_type == "resnet50":
        # Load pretrained ResNet50
        resnet = models.resnet50(
            pretrained=True,
        )
    

    '''
    #To visualize the layers
    children_counter = 0
    for n, c in resnet.named_children():
        print("Children Counter: ", children_counter, " Layer Name: ", n)
        children_counter += 1
    '''

    # Set model to eval mode
    resnet.eval()



    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(input_img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
    # print(img_tensor.shape)
    # Forward pass

    # Pass the image through the model to obtain the features from layer 4
    with torch.no_grad():
        x = img_tensor
        for name, layer in resnet.named_children():
            x = layer(x)
            if name == "layer2":
                layer_features = x
                break
    
    # print(layer_features.shape)
    return layer_features