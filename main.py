# Packages
import torch
import nn

from Models import Discriminator
from Models import Generator

import Functions as F

# Random seed
torch.manual_seed(123)

# Data load

# Models load
discriminator = Discriminator()

generator = Generator()

# Parameters initialization
lr = 0.0001
num_epochs = 25
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# Training the models
F.Train(num_epochs, discriminator, optimizer_discriminator, generator, 
          optimizer_generator)

# Generate a picture
