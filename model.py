import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from model_utils import Discriminator, Generator
from matplotlib import style
from torch.utils.tensorboard import SummaryWriter
style.use("ggplot")

def generate_real_samples(n):
    ''' 1-D Function y = x^2 '''
    # Generate inputs in [-0.5, 0.5]
    X1 = np.random.rand(n) - 0.5
    # Generate outputs X^2
    X2 = X1 * X1
    # Stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = torch.from_numpy(np.hstack((X1, X2))).float()
    # Generate class labels
    y = torch.ones(n, 1) * 0.9
    return X, y

def generate_fake_samples(n):
    ''' 1-D Function y = x^2 '''
    # Generate inputs in [-1, 1]
    X1 = -1 + np.random.rand(n) * 2
    # Generate outputs in [-1, 1]
    X2 = -1 + np.random.rand(n) * 2
    # Stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = torch.from_numpy(np.hstack((X1, X2))).float()
    # Generate class labels
    y = torch.zeros(n, 1) * 0.1
    return X, y

def generate_latent_points(latent_dim, n):
    ''' Generate random latent noise '''
    x_input = torch.from_numpy(np.random.randn(latent_dim * n).reshape(n, latent_dim)).float()
    return x_input

def performance_plot(model, latent_dim, n, name):
    ''' Plotting the generated samples '''
    plt.clf()
    # Ground truth
    x_real, y_real = generate_real_samples(n)
    # Generated points
    x_input = generate_latent_points(latent_dim, n).to(device)
    X = model(x_input).cpu().data.numpy()
    # Plotting the results
    plt.scatter(x_real[:,0], x_real[:,1], color='green')
    plt.scatter(X[:,0], X[:,1], color='red')
    plt.savefig(f'runs/epoch-{name}.png')
   
# Hyperparameters
EPOCHS = 20000
lr = 0.0001
batch_size = 128
n_inputs = 2
latent_dim = 5

# Choosing GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {str(device)}')

# Creating the Discriminator
netD = Discriminator(n_inputs).to(device)
netG = Generator(latent_dim).to(device)

# Setting optimizers
optimD = optim.Adam(netD.parameters(), lr=lr)
optimG = optim.Adam(netG.parameters(), lr=lr)

# Setting loss functions
criterion = nn.BCELoss().to(device)

# Training
netD.train()
netG.train()

half_batch = batch_size // 2
for epoch in range(EPOCHS):
    # Generating real examples
    X_real, y_real = generate_real_samples(half_batch)
    X_real = X_real.to(device)
    y_real = y_real.to(device)
    #print(f'Current device: {torch.cuda.current_device()}, x_real: {X_real.get_device()}, y_real: {y_real.get_device()}')

    # Training the discriminator using real images
    output = netD(X_real).reshape(-1)
    lossD_real = criterion(output, y_real)
    #print(f'Grad: {netD.net[0].weight.grad}')
    D_x = output.mean().item() # the mean confidence
   
    # Generating fake examples
    noise = generate_latent_points(latent_dim, half_batch).to(device)
    x_fake = netG(noise) # Using the generator to produce the fake examples
    y_fake = (torch.zeros(half_batch) * 0.1).to(device)
   
    # Training the discriminator using fake images
    output = netD(x_fake.detach()).reshape(-1)
    lossD_fake = criterion(output, y_fake)
    lossD = lossD_real + lossD_fake
   
    # Updating discriminator's weights
    optimD.zero_grad()
    lossD.backward()
    optimD.step()
   
    # Training the generator
    label = (torch.ones(half_batch)).to(device) # only using half a batch to train the generator
    output = netD(x_fake).reshape(-1)
    lossG = criterion(output, label)
   
    # Updating generator's weights
    optimG.zero_grad()
    lossG.backward()
    optimG.step()
   
    # Visualizing the images generated - Tensorboard
    if epoch % 500 == 0:
        with torch.no_grad():
            performance_plot(netG, latent_dim, 100, epoch)
    print(f'Epoch: [{epoch}/{EPOCHS}], D_Loss: {lossD}, G_Loss: {lossG}, D_x: {D_x}')
   
# Visualizing how the generator learnt
performance_plot(netG, latent_dim, 100, 'final')
