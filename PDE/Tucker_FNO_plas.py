import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam
from tqdm import tqdm

import einops
import math

import os
import scipy.io as sio

from torch.utils.tensorboard import SummaryWriter  



torch.manual_seed(0)
np.random.seed(0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--rank", type=int, default=32)
args = parser.parse_args()


################################################################
# 3d fourier layers
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, rank, s1, s2, s3):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.s1, self.s2, self.s3 = s1, s2, s3 # spatial size

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.rank = rank
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, 8)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.nnx = nn.Conv1d(8*s2*(s3+self.padding), self.width, 1)
        self.nny = nn.Conv1d(8*s1*(s3+self.padding), self.width, 1)
        self.nnz = nn.Conv1d(8*s1*s2, self.width, 1)

        
        # x
        self.convx0 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convx1 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convx2 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convx3 = SpectralConv1d(self.width, self.rank, self.modes3)
        self.wx0 = nn.Conv1d(self.width, self.width, 1)
        self.wx1 = nn.Conv1d(self.width, self.width, 1)
        self.wx2 = nn.Conv1d(self.width, self.width, 1)
        self.wx3 = nn.Conv1d(self.width, self.rank, 1)
        
        # y
        self.convy0 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convy1 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convy2 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convy3 = SpectralConv1d(self.width, self.rank, self.modes3)
        self.wy0 = nn.Conv1d(self.width, self.width, 1)
        self.wy1 = nn.Conv1d(self.width, self.width, 1)
        self.wy2 = nn.Conv1d(self.width, self.width, 1)
        self.wy3 = nn.Conv1d(self.width, self.rank, 1)
        
        # z
        self.convz0 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convz1 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convz2 = SpectralConv1d(self.width, self.width, self.modes3)
        self.convz3 = SpectralConv1d(self.width, self.rank, self.modes3)
        self.wz0 = nn.Conv1d(self.width, self.width, 1)
        self.wz1 = nn.Conv1d(self.width, self.width, 1)
        self.wz2 = nn.Conv1d(self.width, self.width, 1)
        self.wz3 = nn.Conv1d(self.width, self.rank, 1)
        
        

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4)

        centre = torch.randn(self.width, self.rank, self.rank, self.rank)
        stdv = 1 / math.sqrt(centre.size(1))
        centre.uniform_(-stdv, stdv) 
        self.centre = nn.Parameter(centre)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x_shape = x.shape # B C X Y T
        xx = self.nnx(einops.rearrange(x, 'b c x y t -> b (c t y) x')) # B C X Y T -> B C*T X Y
        xy = self.nny(einops.rearrange(x, 'b c x y t -> b (c x t) y')) # B C X Y T -> B C*X*Y T
        xz = self.nnz(einops.rearrange(x, 'b c x y t -> b (c x y) t')) # B C X Y T -> B C*X*Y T


        # ----x convolution layers----
        x1 = self.convx0(xx)
        x2 = self.wx0(xx)
        xx = x1 + x2
        xx = F.gelu(xx)

        x1 = self.convx1(xx)
        x2 = self.wx1(xx)
        xx = x1 + x2
        xx = F.gelu(xx)

        x1 = self.convx2(xx)
        x2 = self.wx2(xx)
        xx = x1 + x2
        xx = F.gelu(xx)

        x1 = self.convx3(xx)
        x2 = self.wx3(xx)
        xx = x1 + x2
        
        # ----y convolution layers----
        x1 = self.convy0(xy)
        x2 = self.wy0(xy)
        xy = x1 + x2
        xy = F.gelu(xy)

        x1 = self.convy1(xy)
        x2 = self.wy1(xy)
        xy = x1 + x2
        xy = F.gelu(xy)

        x1 = self.convy2(xy)
        x2 = self.wy2(xy)
        xy = x1 + x2
        xy = F.gelu(xy)

        x1 = self.convy3(xy)
        x2 = self.wy3(xy)
        xy = x1 + x2
        
        # ----y convolution layers----
        x1 = self.convz0(xz)
        x2 = self.wz0(xz)
        xz = x1 + x2
        xz = F.gelu(xz)

        x1 = self.convz1(xz)
        x2 = self.wz1(xz)
        xz = x1 + x2
        xz = F.gelu(xz)

        x1 = self.convz2(xz)
        x2 = self.wz2(xz)
        xz = x1 + x2
        xz = F.gelu(xz)

        x1 = self.convz3(xz)
        x2 = self.wz3(xz)
        xz = x1 + x2
        
        x = einops.einsum(self.centre, xx, xy, xz, "d i j k, b i n, b j m, b k t -> b d n m t")

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################

ntrain = 1000
ntest = 200

modes = args.modes
width = args.width
rank = args.rank

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()


sub = 1
S = 64 // sub
T_in = 10
T = 40

out_dir = f'./result_abla/plas_rank/tuckerfno3d'
writer = SummaryWriter(os.path.join(out_dir))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

################################################################
# load data
################################################################
N = 987
ntrain = 900
ntest = 80

s1 = 101
s2 = 31
t = 20

r1 = 1
r2 = 1
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

DATA_PATH = './data/plas_N987_T20.mat'

data = sio.loadmat(DATA_PATH)
data['input'] = torch.from_numpy(data['input']).float()
data['output'] = torch.from_numpy(data['output']).float()

x_train = data['input'][:ntrain, ::r1][:, :s1].reshape(ntrain,s1,1,1,1).repeat(1,1,s2,t,1)
y_train = data['output'][:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = data['input'][-ntest:, ::r1][:, :s1].reshape(ntest,s1,1,1,1).repeat(1,1,s2,t,1)
y_test = data['output'][-ntest:, ::r1, ::r2][:, :s1, :s2]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width, rank, s1, s2, t).cuda()

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

t00 = default_timer()

myloss = LpLoss(size_average=False)
for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).view(batch_size, s1, s2, t,4)

        mse = F.mse_loss(out, y, reduction='mean')

        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).view(batch_size, s1, s2, t,4)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    writer.add_scalar('train_loss', train_l2, global_step=ep, walltime=None)
    writer.add_scalar('test_loss', test_l2, global_step=ep, walltime=None)
    writer.add_scalar('time', t2-t1, global_step=ep, walltime=None)

    print(f"epoch: {ep} | time: {t2-t1:.4f} | MSE(train): {train_mse:.4f} | L2(train): {train_l2:.4f} | L2(test): {test_l2:.4f}")

