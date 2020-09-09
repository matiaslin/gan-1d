import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_inputs=2):
        super(Discriminator, self).__init__()
        # Declaring linear hidden layer with He activation
        self.linear = nn.Linear(n_inputs, 25)
        nn.init.kaiming_uniform_(self.linear.weight)
        self.net = nn.Sequential(
            self.linear,
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )
   
    def forward(self, x):
        return self.net(x)
   
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # Declaring linear hidden layer with He activation
        self.linear = nn.Linear(latent_dim, 15)
        nn.init.kaiming_uniform_(self.linear.weight)
        self.net = nn.Sequential(
            self.linear,
            nn.ReLU(),
            nn.Linear(15, 2),
            nn.Tanh()
        )
       
    def forward(self, x):
        return self.net(x)
