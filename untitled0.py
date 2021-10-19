from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(65536,32768),
            nn.ReLu(),
            nn.Dropout(0.3),
            nn.Linear(32768,8192),
            nn.ReLu(),
            nn.Dropout(0.3),
            nn.Linear(8192,2048),
            nn.ReLu(),
            nn.Dropout(0.3),
            nn.Linear(2048,512),
            nn.ReLu(),
            nn.Dropout(0.3),
            nn.Linear(512,1),
            nn.Sigmoid(),)
            
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(8192,16384),
            nn.ReLu(),
            nn.Linear(16384,32768),
            nn.ReLu(),
            nn.Linear(32768,65536),
            nn.Tanh(),
        )
        