import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import torch.nn.init
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio,normalized_root_mse
import time
import random
import os

import scipy.ndimage

from utils import * 
from torch.utils.tensorboard import SummaryWriter
import logging


def main():
    seed=1
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_all =["fruits"]
    g_all = ["0.1"]
    lr = "0.001"
    max_iter = 25001
    k = 12
    width = 32

    for data in data_all:
        for g in g_all: 
                lr_real = float(lr)
                k = int(k)
                width = int(width)
                down = 1
                down_t = 2

                base_dir = os.path.join('tucker_FNO',data,str(g),f"{data}_{g}_{lr_real}_{down_t}")
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)    
                log_dir = os.path.join(base_dir, 'tensorboard_logs')
                writer = SummaryWriter(log_dir)  
                log_file = os.path.join(base_dir, 'train_log.txt')
                print(log_file)
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s',filemode='w')
                logging.info(f"Data: {data}, noise: {g} ,lr:{lr_real}")
                
                obs_file = os.path.join('data', 'Observed', f"{data}-g_{g}.mat")

                mat = scipy.io.loadmat(obs_file)
                X_np = mat["Nhsi"]
                X = torch.from_numpy(X_np).type(dtype).cuda()
                [n_1,n_2,n_3] = X.shape
                
                thres = 0.5
                
                r_1 = int(n_1/down) 
                r_2 = int(n_2/down) 
                r_3 = int(n_3/down_t) 
                print(r_3)

                gt_file = os.path.join('data', 'gt', f"{data}gt.mat")
                mat = scipy.io.loadmat(gt_file)
                gt_np = mat["Ohsi"]
                gt = torch.from_numpy(gt_np).type(dtype).cuda()
    
                params = []

                num_sine = 4
                U_input = torch.from_numpy(np.array(get_cos_sine_mgrid(n_1,num_sine,noise_channel=False))).type(dtype)
                V_input = torch.from_numpy(np.array(get_cos_sine_mgrid(n_2,num_sine,noise_channel=False))).type(dtype)
                model = Tucker_FNO(r_1, r_2, r_3, k, width, n_3).cuda()
                
                optimizier = optim.AdamW([{'params':model.parameters(), 'lr':lr_real},
                                        {'params':params, 'lr':lr_real}]) 
                parameters = sum([p.numel() for p in model.parameters() if p.requires_grad]) + sum([p.numel() for p in params if p.requires_grad])
                print(f'parameters: {parameters}')
                logging.info(f"parameters: {parameters}")
                
                ps_best = 0
                ssim_best = 0
                nmse_best = 0
                cnt = 0

                t0 = time.time()
                for iter in range(max_iter):
                    X_Out = model(U_input,V_input)  
                    loss = torch.norm(X_Out-X,2) # gauss

                    optimizier.zero_grad()
                    loss.backward()
                    optimizier.step()

                    mat_file = os.path.join(base_dir, f'{data}-{g}-denoise.mat')

                    if iter % 100 == 0:
                        ps = peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(),0,1),X_Out.cpu().detach().numpy())
                        ssim = structural_similarity(np.clip(gt.cpu().detach().numpy(),0,1),X_Out.cpu().detach().numpy())
                        nmse = normalized_root_mse(np.clip(gt.cpu().detach().numpy(),0,1),X_Out.cpu().detach().numpy())
                        if ps>ps_best:
                            cnt = 0
                            ps_best = ps
                            ssim_best = ssim
                            nmse_best = nmse
                        else:
                            cnt = cnt + 100

                        print('iteration:',iter,'PSNR',ps,'PSNR_best',ps_best,'SSIM',ssim,'SSIM_best',ssim_best,'NMSE',nmse,'NMSE_best',nmse_best)  

                        writer.add_scalar('PSNR', ps, iter)
                        writer.add_scalar('PSNR_best', ps_best, iter)
                        writer.add_scalar('SSIM', ssim, iter)
                        writer.add_scalar('SSIM_best', ssim_best, iter)
                        writer.add_scalar('NMSE', nmse, iter)
                        writer.add_scalar('NMSE_best', nmse_best, iter)
                        logging.info(f"Iter: {iter}, psnr: {ps:2f}, ps_best: {ps_best:2f}") 
                        logging.info(f"Iter: {iter}, ssim: {ssim:2f}, ssim_best: {ssim_best:2f}")
                        logging.info(f"Iter: {iter}, nmse: {nmse:2f}, nmse_best: {nmse_best:2f}")

                    if cnt >=3000:
                        t1 = time.time()-t0
                        logging.info(f"Time: {t1}, Time/iter: {t1/iter}")
                        logging.info(f"ps_best: {ps_best:2f}, ssim_best: {ssim_best:2f},nmse_best: {nmse_best:2f}")
                        break

                t1 = time.time()-t0
                logging.info(f"Time: {t1}, Time/iter: {t1/max_iter}")
                logging.info(f"ps_best: {ps_best:2f}, ssim_best: {ssim_best:2f},nmse_best: {nmse_best:2f}")
                    
                   
main()