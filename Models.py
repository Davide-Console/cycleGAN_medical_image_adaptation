# Packages
import torch
from torch import nn
import torch.nn.functional as F



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__() # parent class initialization
        
        ### Structure
        
        # For each convolutional layer it is defined:
            # 1: input channels (image z dimension)
            # 2: output channels (image z dimension)
            # 3: kernel size (nxn)
        # (x, y) image dimensions are defined as:
            # x_out = ((x_in+(2*padding)-(dilation*(kernel_size-1)))/stride)+1
            # y_out = ((y_in+(2*padding)-(dilation*(kernel_size-1)))/stride)+1
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 8, 3)
        
        # For each linear layer it is defined:
            # 1: input layers
            # 2: output value (binary)
        self.linear = nn.Linear(8, 1)
        
        # For each pooling layer it is defined:
            # 1: kernel size (nxn)
            # 2: stride (kernel shift)
        self.pool12 = nn.MaxPool2d(5, 2)
        self.pool34 = nn.MaxPool2d(3, 2)
        
        # regularization during train (prevent overfitting)
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        x = self.dropout(self.pool12(F.relu(self.conv1(x))))
        x = self.dropout(self.pool12(F.relu(self.conv2(x))))
        x = self.dropout(self.pool34(F.relu(self.conv3(x))))
        x = self.dropout(self.pool34(F.relu(self.conv4(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.sigmoid(self.linear(x))
        return x
        

    
class Generator(nn.Module):
    def __init__(self):
        super().__init__() # parent class initialization
        
        ### Structure
        
        # For each convolutional layer it is defined:
            # 1: input channels (z dimension)
            # 2: output channels (z dimension)
            # 3: kernel size (nxn)
        # (x, y) image dimensions are defined as:
            # x_out = ((x_in+(2*padding)-(dilation*(kernel_size-1)))/stride)+1
            # y_out = ((y_in+(2*padding)-(dilation*(kernel_size-1)))/stride)+1
        # For each batch normalization layer it is defined:
            # 1: channels (z dimension)
        self.trans1 = nn.ConvTranspose2d(100, 1024, 3)
        self.batchnorm1 = nn.BatchNorm2d(1024)
        self.trans2 = nn.ConvTranspose2d(1024, 512, 3)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.trans3 = nn.ConvTranspose2d(512, 256, 3)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.trans4 = nn.ConvTranspose2d(256, 128, 3)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.trans5 = nn.ConvTranspose2d(128, 1, 5)
        
        # regularization during training (prevent overfitting)
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        x = self.dropout(self.batchnorm1(F.relu(self.trans1(x))))
        x = self.dropout(self.batchnorm2(F.relu(self.trans2(x))))
        x = self.dropout(self.batchnorm4(F.relu(self.trans3(x))))
        x = self.dropout(self.batchnorm4(F.relu(self.trans4(x))))
        x = F.tanh(self.trans5(x))
        


