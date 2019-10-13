### Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
from torch import autograd
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
### Set Parameteres
on_server = 1

train_num = 100000
test_num = 100
batch_size = 200
Max_snr = 30  # Max SNR
min_snr = -10  # min SNR
log_interval = 100
GS_temp = 5 # Temperature
learning_rate = 2e-4
weight_decay = 1e-4
Train_epoch = 10
Test_epoch = 100

if on_server:
    adr ='/nas/ei/home/ga87son/py_sim/AE-GS/'
else:
    adr ='./'

saved_constellation = np.genfromtxt(adr + 'C_128.csv')
Constellation = np.array(saved_constellation)
Const = torch.from_numpy(Constellation)  # .unsqueeze(0)

channel_dim = 2  # number of channels (n)
input_dim = Const.shape[0]  # 2**k, where k = number of bits

temp_min = 0.5
ANNEAL_RATE = 0.003

### Make DataSet
train_snr = torch.rand(train_num) * (Max_snr - min_snr) + min_snr
test_snr = torch.rand(test_num) * (Max_snr - min_snr) + min_snr
test_snr = test_snr.sort()[0]

dataset = Data.TensorDataset(train_snr, train_snr)
datasettest = Data.TensorDataset(test_snr, test_snr)

train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=datasettest, batch_size=1, shuffle=False, num_workers=2)




def sample_gumbel(shape, eps=1e-20):
    """
    Return a sample (or samples) from the "Gumbel" distribution.
    :param shape: torch.Size
    :param eps: float Epsilon, to avoid nan
    :return: Sample(s) from Gumbel distribution
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """
    Return sample from Gumbel-softmax distribution
    :param logits: log of pmf, torch.Size([batch_num, input_dim])
    :param temperature: postitive number, small temperatures tend to one-hot and large variance of the gradients,
    and large temperatures tend to smooth but small variance of the gradients.
    :return: sample from Gumbel-softmax distribution that interpolates between discrete one-hot-encoded categorical distributions and continuous categorical densities
    """
    y = logits + sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)
    return y


def gumbel_softmax(logits, temperature, hard=False):
    """
    Sample from Gumbel-softmax distribution
    :param logits: log of pmf, torch.Size([batch_num, input_dim])
    :param temperature: postitive number, usually set to 1
    :param hard: indicator for soft or hard (one-hot vec.) pmf (Straight-through estimator)
    :return: Gumbel-softmax sample(s)
    """

    y = gumbel_softmax_sample(logits, temperature)

    # return soft pmf
    if not hard:
        return y

    # return on-hot vector in forward path and soft pmf in backward path.
    shape = y.size()
    ind = y.argmax(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class GAE(nn.Module):

    def __init__(self):
        super(GAE, self).__init__()


        self.fc1 = nn.Linear(1, 2 * input_dim)
        self.fc2 = nn.Linear(2 * input_dim, input_dim)

        self.fc3 = nn.Linear(channel_dim + 1, 2 * input_dim)
        self.fc4 = nn.Linear(2 * input_dim, 2 * input_dim)
        self.fc5 = nn.Linear(2 * input_dim, input_dim)

        self.fc6 = nn.Linear(1, 2 * input_dim)
        self.fc7 = nn.Linear(2 * input_dim, input_dim * channel_dim)
        self.fc8 = nn.Linear(2 * input_dim, input_dim * channel_dim)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(.1)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

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
        ebno = 10 ** (ebnodb / 10)
        return ebno

    def forward(self, x):

        logits = self.encode(x.view(-1, 1))     # log of pmf

        p_h = gumbel_softmax(logits, GS_temp, True)   # pmf One-Hot-vector (-1,input_dim)
        p_s = gumbel_softmax(logits, GS_temp, False)  # pmf Soft-vector(-1,input_dim)
        p = F.softmax(logits + 1e-20, dim=-1)   # pmf for normalization

        ## Making the constellation with size (-1,input_dim, channel_dim), like (-1,16,2)
        # for learning Constellation:
        #C = self.make_const(x.view(-1, 1)).view(-1, input_dim, channel_dim)
        # For fixed Constellation
        C = Const.repeat(p_h.shape[0], 1, 1).float()


        # Calculating the weighted norm of constellation, with weghts equal to p
        PC = torch.norm(C, dim=2)  # (-1, input_dim), power of each signal point
        ps = torch.pow(p + 1e-20, 0.5) # size (-1,input_dim)
        Cp = PC*ps
        Cp1 = torch.norm(Cp, dim=-1)
        Cn = C / Cp1.unsqueeze(1).unsqueeze(2) # Normalization



        T = torch.sum(Cn * p_h.unsqueeze(2), dim=1) # Transmitted signal with size (-1,channel_dim)

        channel_snr = 10 ** (x.float() / 10)
        channel_rate = np.log2(input_dim) / channel_dim
        noise = Variable(torch.randn(*T.size()) / ((2 * channel_rate * channel_snr) ** 0.5).view(-1, 1))

        R = T + noise # Received signal with size (-1,channel_dim)

        D = self.decode(torch.cat((R, x.unsqueeze(1)), 1))

        return p_h, D, p, Cn


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        with autograd.detect_anomaly():
            x, x_hat, p, _ = model(data)
            loss = loss_function(x_hat, x, p)
            loss.backward()
        train_loss += loss.item() * len(data)   # Accumulate losses
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
    _MI = np.zeros(len(test_loader.dataset))
    H = np.zeros(len(test_loader.dataset))
    for i, (data, _) in enumerate(test_loader):
        x, x_hat, p, _ = model(data)

        log_ratio = torch.log2(p + 1e-20)
        H[i] = -torch.sum(p * log_ratio, dim=-1).mean().detach().numpy()

        _MI[i] = loss_function(x_hat, x, p).item()
        test_loss += _MI[i] * len(data)

    test_loss /= len(test_loader.dataset)
    print('====> Test {} set loss: {:.4f}'.format(epoch,test_loss))


    return -_MI, H, H+_MI


def plot_shaped_constellation(snr=1):
    test_snr = Variable(torch.tensor([snr]).float())
    T, R, p, C = model(test_snr)
    plot_data = C.data.numpy()
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(plot_data[0, :, 0], plot_data[0, :, 1], np.mean(p.data.numpy(), axis=0) * 2000)
    plt.axis((-3, 3, -3, 3))
    plt.grid()
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=90)
    plt.title('For SNR = '+str(snr)+' dB')
    plt.savefig(adr + 'const_snr_'+str(snr)+'.png')
    plt.show()


def plot_MI_H_CE (MI, H, CE):
    MIa = np.mean(MI,axis=1)
    Ha = np.mean(H,axis=1)
    CEa = np.mean(CE,axis=1)
    plt.figure()
    plt.plot(test_snr.numpy(), MIa)
    plt.title('MI')
    plt.savefig(adr + 'MI.png')
    plt.figure()
    plt.plot(test_snr.numpy(), Ha)
    plt.title('Entropy')
    plt.savefig(adr + 'En.png')
    plt.figure()
    plt.plot(test_snr.numpy(), CEa)
    plt.title('CE')
    plt.savefig(adr + 'CE.png')

def run(Train_epoch = 10, Test_epoch = 100):

    for epoch in range(1, Train_epoch):
        train(epoch)

    MI = np.zeros((test_snr.numpy().shape[0],Test_epoch))
    H = np.zeros_like(MI)
    CE = np.zeros_like(MI)
    for epoch in range(1, Test_epoch):
        MI[:,epoch], H[:,epoch], CE[:,epoch] = test(epoch)

    plot_MI_H_CE(MI, H, CE)
    plot_shaped_constellation(snr=-5)
    plot_shaped_constellation(snr=0.5)
    plot_shaped_constellation(snr=2.5)
    plot_shaped_constellation(snr=5)
    plot_shaped_constellation(snr=10)

def loss_function(x_hat, x, p):
    # Categorical Cross Entropy
    labels = torch.argmax(x, 1)
    loss = nn.NLLLoss()
    CCE = loss(torch.log(x_hat + 1e-20), labels)

    # Source Entropy
    log_ratio = torch.log(p + 1e-20)
    H = -torch.sum(p * log_ratio, dim=-1).mean()

    return (CCE - H)/np.log(2)


model = GAE()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

run(Train_epoch, Test_epoch)

