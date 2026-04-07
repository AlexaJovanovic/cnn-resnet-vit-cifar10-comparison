import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, c_in, height, width, n_classes):
        super().__init__()
        K = 3
        P = 1

        self.conv1 = nn.Conv2d(c_in, 16, K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 16, K, padding=P, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool12 = nn.MaxPool2d(2, 2)
        
        height = (height - 2*K + 4*P + 2) // 2
        width = (width - 2*K + 4*P + 2) // 2
    
        self.conv3 = nn.Conv2d(16, 32, K, padding=P, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 32, K, padding=P, bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        self.pool34 = nn.MaxPool2d(2, 2)
        
        height = (height - 2*K + 4*P + 2) // 2
        width = (width - 2*K + 4*P + 2) // 2
         
        self.n_features = 32 * height * width
        
        self.lin1 = nn.Linear(self.n_features, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, n_classes)   
        
    def forward(self, x):
        # 2x ([conv bn relu]x2 + pool) blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool12(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool34(x)

        # Flattening for FC layers
        x = x.view(-1, self.n_features)
        
        # FC1 and FC2 with reLU activation
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        
        # FC32 for creating logits
        x = self.lin3(x)
        
        return x