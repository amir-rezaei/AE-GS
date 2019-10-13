## Import Libraries
import torch
from torch import nn
import numpy as np
import pdb
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt

## Running on Server?
on_server = 1
if on_server:
    adr ='/nas/ei/home/ga87son/py_sim/AE-GS/'
else:
    adr ='./'

## Set Parameters
num_epoch = 250
batch_size = 500
learning_rate = 0.005
n = 2
k = 10
M = 2**k   #one-hot coding feature dim
channel_dim = n
R = k / channel_dim
input_dim = M

train_num = 10000
test_num = 1000


## Make Dataset
train_labels = (torch.rand(train_num) * M).long()
train_data = torch.sparse.torch.eye(M).index_select(dim=0, index=train_labels)
test_labels = (torch.rand(test_num) * M).long()
test_data = torch.sparse.torch.eye(M).index_select(dim=0, index=test_labels)

# DataBase in Pytorch
dataset = Data.TensorDataset(train_data, train_labels)
datasettest = Data.TensorDataset(test_data, test_labels)
train_loader = Data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
test_loader = Data.DataLoader(dataset =  datasettest, batch_size =  test_num, shuffle = True, num_workers = 2)


class AE(nn.Module):
    def __init__(self, Input_dim, channel_dim):
        super(AE, self).__init__()

        self.Input_dim = Input_dim
        self.channel_dim = channel_dim

        self.encoder = nn.Sequential(
            nn.Linear(Input_dim, Input_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(Input_dim, channel_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(channel_dim, Input_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(Input_dim, Input_dim),
            #nn.Softmax(dim=1)
        )

    def decode_signal(self, x):
        return self.decoder(x)

    def encode_signal(self, x):
        return self.encoder(x)

    def AWGN(self, x, ebno):
        x = (1 / self.channel_dim ** 0.5) * (x / x.std(dim=0))
        communication_rate = R
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * ebno) ** 0.5))
        x += noise
        return x

    def forward(self, x):
        # Encoder:
        x = self.encoder(x)
        # Channel:
        x = (1 / self.channel_dim ** 0.5) * (x / x.std(dim=0))
        training_signal_noise_ratio = 5.01187  # 7dBW to SNR.
        communication_rate = R
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        x += noise
        # Decoder:
        x = self.decoder(x)
        return x


model = AE(M, channel_dim=channel_dim)
loss_fn = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epoch):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_label = Variable(y)
        decoded = model(b_x)
        loss = loss_fn(nn.functional.log_softmax(decoded,dim=-1), b_label)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()        # backpropagation, compute gradients
        optimizer.step()       # apply gradients
        if step % 1000 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)

test_labels = torch.linspace(0, M-1, steps=M).long()
test_data = torch.sparse.torch.eye(M).index_select(dim=0, index=test_labels)
test_data = Variable(test_data)
x = model.encode_signal(test_data)
x = (1 / channel_dim ** 0.5) * (x / x.std(dim=0))
plot_data=x.data.numpy()
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(1*plot_data[:,0],1*plot_data[:,1],c='k')
ax.axis((-2,2,-2,2))
ax.grid()
plt.xlabel("$Re.$", fontsize=18)
plt.ylabel("$Imag.$", fontsize=18, rotation=90)

#for i, txt in enumerate(n):
#    ax.annotate(txt, (1.2*plot_data[i,0],1.2*plot_data[i,1]), fontsize=30)
plt.savefig(adr + 'const_'+str(M)+'.png')
#plt.show()
np.savetxt(adr+'C_'+str(M)+'.csv', plot_data)
