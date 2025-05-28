import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        
        # Load ResNet50 with Weights, and Exclude Last 2 Layers
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2] # Excluding the last two layers (pool and fc)
        self.backbone = nn.Sequential(*modules)
        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Backbone outputs size (Batch, 2048, S, S)
        
        # YOLO Detection Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048*S*S, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=S*S*(C+(B*5)))
        )
        # Head outputs size (Batch x (SxSx(C+Bx5)))
        
    # The value of S (grid size) is directly tied to the input image size and the downsampling done by the backbone.
    # ResNet50 downsamples the input by a factor of 32.
    # So, for an input image of 224x224 → output feature map is 224 / 32 = 7 → S = 7.
    # If you change the input image size (e.g., to 448x448), the output will be 14x14 → S = 14.
    # Make sure S in the YOLO head matches the spatial dimensions of the backbone’s output.
    # Otherwise, the Linear layer dimensions will not match and you'll get shape mismatch errors.
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
    

        
        