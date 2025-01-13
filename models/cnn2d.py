import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, inp_channels = 234, num_class=2):
        super(CNN2D, self).__init__()
        # Input: 64 x 64
        
        # Convolution 1: 3 x 3 @ 32
        self.conv1 = nn.Conv2d(in_channels=inp_channels, out_channels=32, kernel_size=3, stride=1, padding=1)  # Maintain spatial size
        
        # Convolution 2: 3 x 3 @ 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Maintain spatial size
        
        # Pooling 1: 2 x 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial size
        
        # Convolution 3: 3 x 3 @ 64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # Maintain spatial size
        
        # Pooling 2: 2 x 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial size
        
        # Convolution 4: 7 x 7 @ 128
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3)  # Maintain spatial size
        
        # Pooling 3: 2 x 2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial size
        
        # Fully-connected layers
        self.fc1 = None
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 
                            out_features = num_class,
                            bias = True)  # Output layer
        
    def forward(self, x):
        # Pass through convolutional layers and pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        
        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(flattened_size, 2048).to(x.device)
            
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
