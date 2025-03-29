import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(CNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(250880, 256) 
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    model = CNN3D(in_channels=1, num_classes=1)
    dummy_input = torch.randn(2, 1,228, 61, 81) 
    output = model(dummy_input)
    print(output.shape) 
