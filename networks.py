import torch
import torch.nn.functional as foo
from torch import nn
from torchsummary import summary


# Generator
class ResBlock(nn.Module):
    """
        ResBlock(f)

        This class creates a residual block with two convolutional layers.

        Parameters
        ----------
        f : int
            The number of filters in the convolutional layers.
    """
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), nn.InstanceNorm2d(f), nn.ReLU(), nn.Conv2d(f, f, 3, 1, 1))
        self.norm = nn.InstanceNorm2d(f)

    def forward(self, x):
        return foo.relu(self.norm(self.conv(x) + x))


class Generator(nn.Module):
    """
        Generator

        This class takes in a number of arguments and creates a generator.

        Parameters
        ----------
        f : int
          The number of filters in the first convolutional layer.
        blocks : int
          The number of residual blocks in the generator.
    """
    def __init__(self, f, blocks):
        super(Generator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(1, f, 7, 1, 0), nn.InstanceNorm2d(f), nn.ReLU(True),
                  nn.Conv2d(f, 2 * f, 3, 2, 1), nn.InstanceNorm2d(2 * f), nn.ReLU(True),
                  nn.Conv2d(2 * f, 4 * f, 3, 2, 1), nn.InstanceNorm2d(4 * f), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4 * f))
        layers.extend([
            nn.ConvTranspose2d(4 * f, 4 * 2 * f, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(2 * f), nn.ReLU(True),
            nn.ConvTranspose2d(2 * f, 4 * f, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(f), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(f, 1, 7, 1, 0),
            nn.Tanh()])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
        Discriminator

        This class takes in a number of arguments and creates a discriminator.

        Parameters
        ----------
        nc : int
            The number of channels in the first convolutional layer.
        ndf : int
            The number of filters.
    """
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15
            nn.Conv2d(ndf * 8, 1, 4, 1, 1)
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # Cuda activation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    nc = 1
    ndf = 32
    f = 32
    blocks = 9

    D_A = Discriminator(nc, ndf).to(device=device)
    D_B = Discriminator(nc, ndf).to(device=device)

    G_A2B = Generator(f, blocks).to(device=device)
    G_B2A = Generator(f, blocks).to(device=device)

    print('\nG_A2B\n')
    print(summary(G_A2B, (1, 256, 256)))
    print('\nG_B2A\n')
    print(summary(G_B2A, (1, 256, 256)))
    print('\nD_A\n')
    print(summary(D_A, (1, 256, 256)))
    print('\nD_B\n')
    print(summary(D_B, (1, 256, 256)))
