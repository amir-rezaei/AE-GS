### Import Libraries
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pdb
from torch.autograd import Variable
import torch.utils.data as Data

from torch.distributions import kl_divergence
from torch.distributions import Categorical
from torch import autograd
import matplotlib.pyplot as plt


### Set Parameteres

train_num  = 2**19
test_num   = 2**10
BATCH_SIZE = 2**8
Msnr       = 40   # Max SNR
msnr       = 0   # min SNR
log_interval = 50

csv = np.genfromtxt('C16.csv')
C = np.array(csv)

Const = torch.from_numpy(C)#.unsqueeze(0)

latent_dim = 2        # Namely, number of channels (n)
categorical_dim = Const.shape[0]  # Namely, 2**k, where k = number of bits

temp_min = 0.5
ANNEAL_RATE = 0.003



### Make DataSet
train_snr = torch.rand(train_num)*(Msnr-msnr) + msnr
test_snr  = torch.rand(test_num)*(Msnr-msnr) + msnr
test_snr = test_snr.sort()[0]

dataset     = Data.TensorDataset(train_snr, train_snr)
datasettest = Data.TensorDataset(test_snr, test_snr)

train_loader = Data.DataLoader(dataset = dataset,     batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
test_loader  = Data.DataLoader(dataset = datasettest, batch_size = 1,   shuffle = False, num_workers = 2)



### Required Functions

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)
    return y


def gumbel_softmax(logits, temperature, hard=False):
    
    y  = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard 


class VAE_gumbel(nn.Module):
    
    def __init__(self):
        super(VAE_gumbel, self).__init__()
        
        self.fc1 = nn.Linear(1, 2*categorical_dim)
        self.fc2 = nn.Linear(2*categorical_dim, categorical_dim)
       
        self.fc3 = nn.Linear(latent_dim + 1, 2*categorical_dim)
        self.fc4 = nn.Linear(2*categorical_dim, 2*categorical_dim)
        self.fc5 = nn.Linear(2*categorical_dim, categorical_dim)

        self.fc6 = nn.Linear(1, 2*categorical_dim)
        self.fc7 = nn.Linear(2*categorical_dim, categorical_dim * latent_dim)
        self.fc8 = nn.Linear(2*categorical_dim, categorical_dim * latent_dim)
        

        
        self.relu    = nn.ReLU()
        self.lrelu   = nn.LeakyReLU(.1)
        self.sigmoid = nn.Sigmoid()
        self.sm      = nn.Softmax(dim=1)

    def encode(self, x):
        h1 = self.lrelu(self.fc1(x))
        return self.fc2(h1)
    
    def decode(self, z):
        h2 = self.lrelu(self.fc3(z))
        h3 = self.lrelu(self.fc4(h2))
        return self.sm(self.fc5(h3))
    
    def make_const(self, x):
        h4 = self.lrelu(self.fc6(x))
        h5 = self.lrelu(self.fc7(h4))
        return self.fc8(h5)
        
    
    
    def EbNo_to_noise(self, ebnodb):
        ebno = 10**(ebnodb/10)
        return noise_std

    def forward(self, x):

        logits = self.encode(x.view(-1, 1))
        
        p_h = gumbel_softmax(logits,1,True)  # Prob. One-Hot-vector (-1,16)
        p_s = gumbel_softmax(logits,1,False) # Prob. Soft-vector(-1,16)
        p  = F.softmax(logits+1e-20, dim=-1)
        
        C = self.make_const(x.view(-1,1)).view(-1, categorical_dim, latent_dim) # for learning Constellation
        #C = Const.repeat(p_h.shape[0], 1, 1).float() # For fixed Constellation

        ps   = torch.pow(p+1e-20,0.5)
        
        
        phi = torch.atan2(C[:,:,1],C[:,:,0])
        rho = torch.norm(C,dim=2)
        
        rhop = rho*ps
        rhopn = torch.norm(rhop,dim=-1)
        rh = rho/rhopn.unsqueeze(1)

        CX = rh * torch.cos(phi)
        CY = rh * torch.sin(phi)
        
        Cn = torch.cat((CX.unsqueeze(2), CY.unsqueeze(2)), 2)
        
        T = torch.sum(Cn*p_h.unsqueeze(2), dim =1)
 
        training_signal_noise_ratio =  10**(x.float()/10)     
        communication_rate = np.log2(categorical_dim)/latent_dim
        noise = Variable(torch.randn(*T.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5).view(-1,1))
        
        R = T + noise 
        D = self.decode(torch.cat((R, x.unsqueeze(1)), 1))
        

    
        return p_h,D,p,Cn


def train(epoch):
    model.train()
    train_loss = 0
    temp = 1
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        with autograd.detect_anomaly():
            p_h,recon_batch, p_s,_ = model(data)
            loss = loss_function(recon_batch, p_h, p_s)
            loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    temp = np.zeros(len(test_loader.dataset))
    H = np.zeros(len(test_loader.dataset))
    for i, (data, _) in enumerate(test_loader):
        st,recon_batch, qy,_ = model(data)
        
        log_ratio = torch.log2(qy + 1e-20)
        H[i] = -torch.sum(qy * log_ratio, dim=-1).mean().detach().numpy()
        
        temp[i] = loss_function(recon_batch, st, qy).item() #- H
        test_loss += temp[i] * len(data)
        
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    plt.figure()
    plt.plot(test_snr.numpy(),-temp)
    plt.title('MI')
    plt.figure()
    plt.plot(test_snr.numpy(),H)
    plt.title('Entropy')
    plt.figure()
    plt.plot(test_snr.numpy(),H+temp)
    plt.title('CE')
    
def run():
    for epoch in range(1, 2):
        plt.show()
        train(epoch)
        test(epoch)
        
        
        
def loss_function(x_hat, x, p):
    
    
    ### Categorical Cross Entropy
    _, labels = torch.max(x,1)
    loss = nn.NLLLoss()
    CCE =  loss(torch.log(x_hat+1e-20),labels)
    
    ### Source Entropy
    log_ratio = torch.log2(p + 1e-20)
    H = -torch.sum(p * log_ratio, dim=-1).mean()
    
    
    return CCE - H
    
    
model = VAE_gumbel()
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)


run()


test_labels = torch.linspace(5, 20, steps=15)
test_labels = torch.tensor([.5]).float()
print(test_labels[0])
test_labels = torch.cat([test_labels]*1)
test_data = Variable(test_labels)
T,R,q,C = model(test_data)
plot_data = C.data.numpy()
fig = plt.figure(figsize=(4,4))
plt.scatter(plot_data[0,:,0],plot_data[0,:,1],np.mean(q.data.numpy(),axis=0)*2000)
plt.axis((-3,3,-3,3))
plt.grid()
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=90)
plt.show()
