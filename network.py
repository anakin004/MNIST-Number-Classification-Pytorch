import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self._init_fc()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 10)

    # dynamic flatten size calc
    def _init_fc(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            self.flattened_size = x.view(1, -1).shape[1]

    def forward(self, x):               # B -> Batch Size
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 7, 7)
        
        x = x.view(x.size(0), -1)                      # flatten -> (B, 64*7*7)
        x = F.relu(self.fc1(x))                        # hidden dense layer
        x = self.dropout(x)                            # apply dropout to generalize
        x = self.fc2(x)                                # output logits
        return x