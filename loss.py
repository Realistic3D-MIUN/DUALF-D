import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN
from parameters import *
import numpy as np
import math

class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
https://www.c3se.chalmers.se/about/C3SE/
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()

################################################ Loss ############################################

import torch
import torch.nn.functional as F
import math
import torch
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
import math

def compute_metrics(out, x):
    """Compute detailed metrics for monitoring"""
    x_hat = out["x_hat"]
    likelihoods = out["likelihoods"]
    
    # Compute MSE and PSNR
    mse = F.mse_loss(x_hat, x).item()
    psnr = -10 * math.log10(mse) if mse > 0 else 100
    
    # Compute bpp for each component
    N, _, H, W = x.size()
    num_pixels = N * H * W
    
    bpp_y_s = -torch.log2(likelihoods["y_s"]).sum().item() / num_pixels
    bpp_y_a = -torch.log2(likelihoods["y_a"]).sum().item() / num_pixels
    bpp_z_s = -torch.log2(likelihoods["z_s"]).sum().item() / num_pixels
    bpp_z_a = -torch.log2(likelihoods["z_a"]).sum().item() / num_pixels
    
    total_bpp = bpp_y_s + bpp_y_a + bpp_z_s + bpp_z_a
    
    return {
        'mse': mse,
        'psnr': psnr,
        'bpp_total': total_bpp,
        'bpp_y_spatial': bpp_y_s,
        'bpp_y_angular': bpp_y_a,
        'bpp_z_spatial': bpp_z_s,
        'bpp_z_angular': bpp_z_a
    }

def dual_hyperprior_loss(out, x, lambda_factor=0.01):
    """Loss function for the dual-hyperprior VAE model"""
    x_hat = out["x_hat"]
    likelihoods = out["likelihoods"]
    
    # Distortion - using sum like your previous network
    mse_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # Rate: both main latents and hyperprior
    N, _, H, W = x.size()
    num_pixels = N * H * W
    
    bpp_y_s = -torch.log2(likelihoods["y_s"]).sum() / num_pixels
    bpp_y_a = -torch.log2(likelihoods["y_a"]).sum() / num_pixels
    bpp_z_s = -torch.log2(likelihoods["z_s"]).sum() / num_pixels
    bpp_z_a = -torch.log2(likelihoods["z_a"]).sum() / num_pixels
    
    bpp = bpp_y_s + bpp_y_a + bpp_z_s + bpp_z_a
    
    return mse_loss + lambda_factor * bpp, mse_loss, bpp

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


import torch
import torch.nn as nn
import torch.nn.functional as F

class DualHyperpriorLoss(nn.Module):
    def __init__(self, lmbda=0.001, beta=0.001, projection_dim=128, temperature=0.1):
        """
        lmbda: weight for rate-distortion tradeoff
        beta: weight for mutual information regularization
        projection_dim: dimension for the projection space
        temperature: temperature parameter for InfoNCE
        """
        super().__init__()
        self.lmbda = lmbda
        self.beta = beta
        self.temperature = temperature
        
        # Initialize projectors in __init__ (handles dynamic adaptation)
        self.proj_s = None
        self.proj_a = None
        self.projection_dim = projection_dim
        self.initialized = False

    def initialize_projectors(self, C_s, C_a, device):
        """Initialize projection layers with proper dimensions"""
        self.proj_s = nn.Linear(C_s, self.projection_dim).to(device)
        self.proj_a = nn.Linear(C_a, self.projection_dim).to(device)
        self.initialized = True

    def compute_mi_loss(self, y_s, y_a):
        """
        Compute mutual information using InfoNCE with positive/negative samples.
        """
        N, C_s, H_s, W_s = y_s.shape
        _, C_a, H_a, W_a = y_a.shape
        
        # Flatten spatial and angular latent representations
        y_s_flat = y_s.view(N, C_s, -1).mean(dim=2)  # [N, C_s]
        y_a_flat = y_a.view(N, C_a, -1).mean(dim=2)  # [N, C_a]
        
        # Initialize projectors if not done yet
        if not self.initialized:
            self.initialize_projectors(C_s, C_a, y_s.device)
        
        # Project to common dimension
        y_s_proj = self.proj_s(y_s_flat)  # [N, projection_dim]
        y_a_proj = self.proj_a(y_a_flat)  # [N, projection_dim]
        
        # Normalize features
        y_s_norm = F.normalize(y_s_proj, dim=1)
        y_a_norm = F.normalize(y_a_proj, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(y_s_norm, y_a_norm.t()) / self.temperature  # [N, N]
        
        # InfoNCE Loss
        labels = torch.arange(N, device=y_s.device)
        loss_s_to_a = F.cross_entropy(sim_matrix, labels)
        loss_a_to_s = F.cross_entropy(sim_matrix.t(), labels)  # Reverse direction
        
        return (loss_s_to_a + loss_a_to_s) / 2.0  # Symmetric MI Loss

    def forward(self, out, x):
        """
        Compute total loss including distortion, rate, and MI regularization.
        """
        N, _, H, W = x.size()
        num_pixels = N * H * W

        # 1. MSE Loss (Distortion)
        mse_loss = F.mse_loss(out["x_hat"], x, reduction='mean')

        # 2. Rate Loss (BPP)
        bpp_loss = sum(
            (-torch.log2(likelihoods).sum()) / num_pixels
            for likelihoods in out["likelihoods"].values()
        )

        # 3. Mutual Information Loss
        mi_loss = torch.tensor(0.0, device=x.device)
        if "latents" in out:
            mi_loss = self.compute_mi_loss(out["latents"]["y_s"], out["latents"]["y_a"])

        # Total loss
        total_loss = mse_loss + self.lmbda * bpp_loss - self.beta * mi_loss

        return total_loss, mse_loss, bpp_loss, mi_loss
