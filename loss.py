import torch
import torch.nn as nn

class YOLOv1_Loss():
    def __init__(self, S=7, B=2, C=20, coord=5, noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord = coord
        self.noobj = noobj
        
        self.mse = nn.MSELoss(reduction='sum')