import torch
from torch import nn
import numpy as np
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pdb
import time
from torchdiffeq import odeint

class Net(torch.nn.Module):
    def __init__(self, widths_encoder, widths_decoder, poly_order, model_order, input_dim, latent_dim, sindy_dim, N, t, device):
        super().__init__()
        self.model_order = model_order
        self.poly_order = poly_order
        self.widths_encoder = widths_encoder
        self.widths_decoder = widths_decoder
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sindy_dim = sindy_dim
        self.encoder = nn.Sequential
        self.decoder = nn.Sequential
        self.device = device
        self.E = []
        self.N = N
        self.t = t
        self.odeint = odeint

        # Sindy Layer
        self.E = nn.Linear(self.sindy_dim, self.latent_dim)
        nn.init.constant_(self.E.weight, 0.5)

        # Build encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.widths_encoder[:-1])):
            # Encoder

            temp = nn.Linear(self.widths_encoder[i], self.widths_encoder[i+1])
            nn.init.xavier_uniform_(temp.weight, gain=1)

            # temp = nn.Identity()

            self.encoder.append(temp)

            if i < len(self.widths_encoder[:-1]) - 1:
                self.encoder.append(nn.Identity())

        # Builder decoder
        self.decoder = nn.ModuleList()
        for i in range(len(self.widths_decoder[:-1])):
            # Encoder
            temp = nn.Linear(self.widths_decoder[i], self.widths_decoder[i+1])
            nn.init.xavier_uniform_(temp.weight, gain=1)

            # temp = nn.Identity()

            self.decoder.append(temp)

            if i < len(self.widths_decoder[:-1]) - 1:
                self.decoder.append(nn.Identity())

    def encoder_forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def decoder_forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def sindy_forward(self, t, z):
        theta = sindy_library(z, self.poly_order, self.latent_dim, self.device)
        dzb = self.E(theta)
        return dzb

    def forward(self, x, dx):

        z = self.encoder_forward(x)

        if self.model_order == 1:
            dz = self.calc_dz(x, dx, 1)
            dzb = self.sindy_forward(self.t, z)

            xb = self.decoder_forward(z)
            dxb = self.calc_dz(z, dzb, 0)

            return z, dz, dzb, xb, dxb

    def calc_gx_enc(self, x):
        N = x.shape[0]
        gx = torch.empty((N, self.latent_dim, self.input_dim), device=self.device)
        for i in range(N):
            gx[i, :, :] = torch.autograd.functional.jacobian(self.encoder_forward, x[i, :], create_graph=True)
        return gx

    def calc_gx_dec(self, x):
        N = x.shape[0]
        gx = torch.empty((N, self.input_dim, self.latent_dim), device=self.device)
        for i in range(N):
            gx[i, :, :] = torch.autograd.functional.jacobian(self.decoder_forward, x[i, :], create_graph=True)
        return gx

    def calc_dz(self, x, dx, is_encoder):
        N = x.shape[0]
        if is_encoder:
            gx = self.calc_gx_enc(x)
            dz = torch.empty((N, self.latent_dim), device=self.device)
        else:
            gx = self.calc_gx_dec(x)
            dz = torch.empty((N, self.input_dim), device=self.device)

        for i in range(N):
            dz[i] = torch.matmul(gx[i], dx[i])

        return dz


def sindy_library(z, poly_order, latent_dim, device):

    if len(z.shape) == 1:
        if device == 'cpu':
            z = np.expand_dims(z, axis=0)
        else:
            z = torch.unsqueeze(z, dim=0)

    N = z.shape[0]
    library = []

    # Constant
    for i in range(latent_dim):
        library.append(torch.ones(N, device=device))

    # 1st order
    if poly_order > 0:
        for i in range(latent_dim):
            library.append(z[:, i])

    # 2nd order
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:, i]*z[:,j])

    # 3rd order
    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    # 4th order
    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    # 5th order
    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for m in range(p, latent_dim):
                            library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, m])

    # 6th order
    if poly_order > 5:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for m in range(p, latent_dim):
                            for q in range(p, latent_dim):
                                library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, m] * z[:, q])

    if device == 'cpu':
        return np.stack(library, axis=1)
    else:
        return torch.stack(library, axis=1)


def sindy_simulate(z, t, poly_order, latent_dim, E):

    theta = sindy_library(z, poly_order, latent_dim, 'cpu')
    E = E.weight.detach().cpu().numpy()
    E[np.where(np.abs(E) < 0.1)] = 0
    dz = np.matmul(theta, E.T)

    return np.squeeze(dz)


# dev = 'cpu'
# input_dim = 64
# N = 128
# latent_dim = 8
# widths = [input_dim, 128, 64, 32, 16, latent_dim]
# poly_order = 4
# model_order = 1
# temp = sindy_library(torch.ones((N, latent_dim), device=dev), poly_order, latent_dim, N)
# sindy_dim = temp.shape[1]
#
# x = torch.rand((N, input_dim))
# dx = torch.rand((N, input_dim))
# ddx = torch.rand((N, input_dim))
#
# net = Net(widths, poly_order, model_order, input_dim, latent_dim, sindy_dim, N)
#
# # tic = time.perf_counter()
# # for i in range(N):
# #     gx = net.calc_gx_enc(x[i, :])
# # toc = time.perf_counter()
# # print(toc-tic)
#
# tic = time.perf_counter()
# gx = net.calc_gx_enc(x)
# dz = net.calc_dz(x, dx, 1)
# pdb.set_trace()
# toc = time.perf_counter()
# print(toc-tic)
#
#
# pdb.set_trace()
