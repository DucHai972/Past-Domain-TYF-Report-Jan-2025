import torch
import torch.nn as nn
from torchsummary import summary

class CNN2D(nn.Module):
    def __init__(self,in_channels = 3, num_classes=1):
        super(CNN2D, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8960, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1) if num_classes > 2 else nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # breakpoint()
        x = self.fc_layers(x)
        return x
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN2D(in_channels=228).to(device)
    
    # x = torch.rand(2,228,64,64)
    print(summary(model,(228,61,81)))
    model.load_state_dict(torch.load("/N/slate/tnn3/HaiND/01-06_report/result/model/trained_model_cnn2d_t2_rus4_cw2.pth"))