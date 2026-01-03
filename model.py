import torch
import torch.nn as nn
from compressai.layers import GDN
from parameters import *
from compressai.entropy_models import GaussianConditional
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import conv3x3, subpel_conv3x3
import math
import torch.nn.functional as F

class SpatialEncoder(nn.Module):
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.conv_layers_S1 = nn.Sequential(
            nn.Conv2d(3, filt_n, kernel_size=5, stride=1, padding=1, dilation=3),
            GDN(filt_n)
        )
        self.conv_layers_S2 = nn.Sequential(
            nn.Conv2d(filt_n, filt_n, kernel_size=5, stride=2, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_S3 = nn.Sequential(
            nn.Conv2d(filt_n, filt_n, kernel_size=5, stride=2, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_S4 = nn.Sequential(
            nn.Conv2d(filt_n, filt_n, kernel_size=5, stride=2, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_S5 = nn.Sequential(
            nn.Conv2d(filt_n, 64, kernel_size=5, stride=3, padding=1),
            GDN(64)
        )

    def forward(self, x):
        x = self.conv_layers_S1(x)
        x = self.conv_layers_S2(x)
        x = self.conv_layers_S3(x)
        x = self.conv_layers_S4(x)
        x = self.conv_layers_S5(x)
        return x

class AngularEncoder(nn.Module):
    def __init__(self):
        super(AngularEncoder, self).__init__()
        self.conv_layers_A1 = nn.Sequential(
            nn.Conv2d(3, filt_n, kernel_size=3, stride=3, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_A2 = nn.Sequential(
            nn.Conv2d(filt_n, filt_n, kernel_size=5, stride=2, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_A3 = nn.Sequential(
            nn.Conv2d(filt_n, filt_n, kernel_size=5, stride=2, padding=1),
            GDN(filt_n)
        )
        self.conv_layers_A4 = nn.Sequential(
            nn.Conv2d(filt_n, 64, kernel_size=5, stride=2, padding=1),
            GDN(64)
        )

    def forward(self, x):
        x = self.conv_layers_A1(x)
        x = self.conv_layers_A2(x)
        x = self.conv_layers_A3(x)
        x = self.conv_layers_A4(x)
        return x

class HyperpriorNetwork(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.h_a = nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(inplace=True),
            conv3x3(channels, channels, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(channels, channels, stride=2),
        )
        
        self.h_s = nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(channels, channels, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(channels, channels, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(channels, channels),
        )

    def forward(self, x):
        # Analysis transform
        z = self.h_a(x)
        
        # Quantization and entropy coding
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Synthesis transform to get scales
        scales = torch.exp(self.h_s(z_hat))  # Added exp to ensure positive scales
        
        return scales, z_likelihoods

    def update(self, force=False):
        """Updates the entropy bottleneck parameters."""
        return self.entropy_bottleneck.update(force=force)

def get_scale_table(min=0.11, max=256, levels=64):
    """Get the scale table from CompressAI."""
    return [float(f) for f in torch.exp(torch.linspace(math.log(min), math.log(max), levels))]

class Encoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.spatial_encoder = SpatialEncoder()
        self.angular_encoder = AngularEncoder()

        scale_table = get_scale_table(min=0.11, max=256, levels=64)
        self.spatial_hyperprior = HyperpriorNetwork(64)
        self.angular_hyperprior = HyperpriorNetwork(64)
        
        self.entropy_model_s = GaussianConditional(scale_table)
        self.entropy_model_a = GaussianConditional(scale_table)

    def forward(self, x):
        # Get latent representations
        y_s = self.spatial_encoder(x)
        y_a = self.angular_encoder(x)

        # Get scales from hyperpriors
        scales_s, z_likelihood_s = self.spatial_hyperprior(y_s)
        scales_a, z_likelihood_a = self.angular_hyperprior(y_a)

        # Quantize latents using gaussian conditional
        z_s, likelihood_s = self.entropy_model_s(y_s, scales_s)
        z_a, likelihood_a = self.entropy_model_a(y_a, scales_a)

        concatenated = torch.cat((z_s, z_a), dim=1)

        return {
            "y_hat": concatenated,
            "latents": {"y_s": y_s, "y_a": y_a},  # Add latents for loss function
            "likelihoods": {
                "y_s": likelihood_s,
                "y_a": likelihood_a,
                "z_s": z_likelihood_s,
                "z_a": z_likelihood_a
            }
        }

class Decoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, filt_n, kernel_size=5, stride=3, padding=0),
            GDN(filt_n, inverse=True)
        )
        self.conv_layers_D1 = nn.Sequential(
            nn.ConvTranspose2d(filt_n, filt_n, kernel_size=4, stride=2, padding=0),
            GDN(filt_n, inverse=True)
        )
        self.conv_layers_D2 = nn.Sequential(
            nn.ConvTranspose2d(filt_n, filt_n, kernel_size=4, stride=2, padding=1),
            GDN(filt_n, inverse=True)
        )
        self.conv_layers_D3 = nn.Sequential(
            nn.ConvTranspose2d(filt_n, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.initial_layer(z)
        x = self.conv_layers_D1(x)
        x = self.conv_layers_D2(x)
        x = self.conv_layers_D3(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out["y_hat"])
        
        return {
            "x_hat": dec_out,
            "likelihoods": enc_out["likelihoods"],
            "latents": enc_out["latents"]  # Pass through the latents
        }

    def aux_loss(self):
        """Return the aggregated auxiliary loss."""
        aux_loss = self.encoder.spatial_hyperprior.entropy_bottleneck.loss()
        aux_loss += self.encoder.angular_hyperprior.entropy_bottleneck.loss()
        return aux_loss

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values."""
        updated = False
        updated |= self.encoder.spatial_hyperprior.entropy_bottleneck.update(force=force)
        updated |= self.encoder.angular_hyperprior.entropy_bottleneck.update(force=force)
        return updated

class DualHyperpriorLoss(nn.Module):
    def __init__(self, lmbda=0.001):
        super().__init__()
        self.lmbda = lmbda

    def forward(self, out, x):
        N, _, H, W = x.size()
        num_pixels = N * H * W

        # Check if values are in [0,1]
        if torch.max(out["x_hat"]) > 1 or torch.min(out["x_hat"]) < 0:
            print(f"x_hat range: [{torch.min(out['x_hat']).item()}, {torch.max(out['x_hat']).item()}]")
        
        # Use sum reduction
        mse_loss = F.mse_loss(out["x_hat"], x, reduction='sum')
        
        # Compute bpp loss
        bpp_loss = sum((-torch.log2(likelihoods).sum())
                      for likelihoods in out["likelihoods"].values()) / num_pixels

        # Calculate PSNR properly
        mse = mse_loss / num_pixels  # Convert sum to mean
        psnr = -10 * torch.log10(mse)

        return mse_loss + self.lmbda * bpp_loss, mse_loss, bpp_loss, psnr
