# Packages
from torch import nn
import torch.nn.functional as foo

# Generators


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), nn.InstanceNorm2d(f), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = nn.InstanceNorm2d(f)

    def forward(self, x):
        return foo.relu(self.norm(self.conv(x)+x))


class Generator(nn.Module):
    def __init__(self, f=64, blocks=6):
        super(Generator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(1, f, 7, 1, 0), nn.InstanceNorm2d(f), nn.ReLU(True),
                  nn.Conv2d(f, 2*f, 3, 2, 1), nn.InstanceNorm2d(2*f), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), nn.InstanceNorm2d(4*f), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
        layers.extend([
                nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(2*f), nn.ReLU(True),
                nn.ConvTranspose2d(2*f, 4*f, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(f), nn.ReLU(True),
                nn.ReflectionPad2d(3), nn.Conv2d(f, 1, 7, 1, 0),
                nn.Tanh()])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)
    
# Discriminator


class Discriminator(nn.Module):  
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15
            nn.Conv2d(ndf*8, 1, 4, 1, 1)
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        return self.main(input)
    
