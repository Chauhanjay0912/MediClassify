import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True, device='cuda'):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Unfreeze last layers for fine-tuning
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Better classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(device)
