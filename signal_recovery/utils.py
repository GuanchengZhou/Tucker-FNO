import torch
import torch.nn.functional as F
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import torch.nn.init
import math
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.init as init
import einops


def get_cos_sine_mgrid(sidelen, num_sine, noise_channel=True, dim=2):
    index = torch.linspace(-1, 1, steps=sidelen)
    emb = []  

    for i in range(num_sine):
        value = torch.sin((2**i) * math.pi * index)
        emb.append(value)
        value = torch.cos((2**i) * math.pi * index)
        emb.append(value)

    value = torch.stack(emb).permute(1,0) 
    value = torch.cat([index.unsqueeze(-1), value], dim=-1)
    return value  

class SineLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True,
				is_first=False, omega_0=0.2):
		super().__init__()
		self.omega_0 = omega_0
		self.is_first = is_first
		self.in_features = in_features
		self.linear = nn.Linear(in_features, out_features, bias=bias)
		self.relu = nn.ReLU()

	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				init.kaiming_uniform_(self.linear.weight, a=np.sqrt(5)) 
			else:
				gain = 1 / self.omega_0
				init.kaiming_uniform_(self.linear.weight, a=np.sqrt(5), mode='fan_in')
				self.linear.weight.data.mul_(gain)
		
	def forward(self, input):
		return torch.sin(self.omega_0*self.linear(input))
	
class FNO_layer(nn.Module):
	def __init__(self, k, d, is_first = False):
		super(FNO_layer, self).__init__()

		self.is_first = is_first
		self.k = k

		self.P_low = nn.Parameter(torch.randn(k, d, d, 2) * (1. / d))
		self.B_low = nn.Parameter(torch.randn(k, d, 2) * (1. / d))
		
		self.W = nn.Parameter(torch.randn(d, d) * (1. / d))
		self.B = nn.Parameter(torch.randn(d) * (1. / d))
		
		self.init_weights()

	def init_weights(self):
		with torch.no_grad():
			if self.is_first:
				init.kaiming_uniform_(self.W, a=np.sqrt(5)) 
			else:
				gain = 1 / 0.2
				init.kaiming_uniform_(self.W, a=np.sqrt(5), mode='fan_in')
				self.W.mul_(gain)
				init.kaiming_uniform_(self.P_low, a=np.sqrt(5), mode='fan_in')
				self.P_low.mul_(gain)
	
	def forward(self, x):
		x_out = torch.einsum('nd,dr->nr', x, self.W) + self.B 
		x_ft = torch.fft.rfftn(x, dim=0, norm='forward')  # [n, d]
		
		x_low = x_ft[:self.k]      # shape: [split, d]
		x_ft_out = torch.zeros_like(x_ft)

		W = torch.view_as_complex(self.P_low)
		B = torch.view_as_complex(self.B_low)

		out_low = torch.einsum('kd,kdf->kf', x_low, W) + B
		x_ft_out[:self.k] = out_low
		x_res = torch.fft.irfftn(x_ft_out, s=(x.shape[0],), dim=0, norm='forward')

		return torch.sin(0.2*(x_out + x_res))

class Tucker_FNO(nn.Module):
    def __init__(self, r_1, r_2, r_3, k, mid_channel, out_ch):
        super(Tucker_FNO,self).__init__()
        print('mid', mid_channel)
        
        self.U_net = nn.Sequential(
            SineLayer(21, mid_channel, is_first=False),
            FNO_layer(k, mid_channel, is_first=True),
            FNO_layer(k, mid_channel, is_first=True),
            nn.Linear(mid_channel, r_1)
        )
        
        self.V_net = nn.Sequential(
            SineLayer(21, mid_channel, is_first=False),
            FNO_layer(k, mid_channel, is_first=True),
            FNO_layer(k, mid_channel, is_first=True),
            nn.Linear(mid_channel, r_2)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(r_3, r_3),
            nn.GELU(),
            nn.Linear(r_3, out_ch),
        )

        centre = torch.zeros((r_1, r_2, r_3)).cuda()
        stdv = 1 / math.sqrt(r_2)  
        centre.uniform_(-stdv, stdv)
        centre.requires_grad = True
        self.centre = nn.Parameter(centre) 

    def forward(self, U_input, V_input):
        U = self.U_net(U_input)  
        V = self.V_net(V_input)  

        centre = einops.einsum(
            self.centre, U, V,
            'r1 r2 c, n1 r1, n2 r2 -> n1 n2 c'
        )
        centre = self.proj(centre)  
        return centre