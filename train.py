import torch
from torch import nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from network import Net, sindy_library, sindy_simulate
from torchdiffeq import odeint

import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control

from scipy.integrate import solve_ivp

import os
import time
import sys
import pdb

use_gpu = False
if use_gpu:
    dev = 'cuda:0'
    device = torch.device(dev)
else:
    dev = 'cpu'
    device = torch.device(dev)

dt = 0.02
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

t_end_train = 10
t_end_test = 15
t_train = np.arange(0, t_end_train, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

# random mixing matrix
m = np.array([[0, 0, 2],
              [3, 0, 2],
              [0, 1, 1],
              [1, 0, 4],
              [1, 0, 0]])

x = x @ m.T

N, input_dim = x.shape

dx = np.zeros(x.shape)
for i in range(input_dim):
    dx[:, i] = ps.SmoothedFiniteDifference()._differentiate(x[:, i], t_train)

x = torch.Tensor(x)
dx = torch.Tensor(dx)

latent_dim = 3
widths_encoder = [input_dim, latent_dim]
widths_decoder = [latent_dim, input_dim]
poly_order = 3
model_order = 1
temp = sindy_library(torch.ones((N, latent_dim), device=dev), poly_order, latent_dim, device)
sindy_dim = temp.shape[1]

net = Net(widths_encoder, widths_decoder, poly_order, model_order, input_dim, latent_dim, sindy_dim, N, torch.tensor(t_train), device)

if use_gpu:
    net = net.to(device)
    x = x.to(device)
    dx = dx.to(device)

epochs = 500
losses = []
losses_unreg = []
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(),
                             lr = 1e-2,
                             )

batch_size = 1
lamdba = 1/100

mask = torch.ones(net.E.weight.shape)

for epoch in range(epochs):

    z, dz, dzb, xb, dxb = net(x, dx)

    zsol = np.zeros(z.shape)

    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].plot(x.detach().numpy())
    # axs[0, 1].plot(z.detach().numpy())
    # axs[0, 2].plot(xb.detach().numpy())
    # axs[1, 0].plot(dx.detach().numpy())
    # axs[1, 1].plot(dz.detach().numpy())
    # axs[1, 2].plot(dzb.detach().numpy())
    # fig.show()
    #
    # pdb.set_trace()

    loss = loss_function(xb, x)/batch_size
    loss += 1/1000*loss_function(dz, dzb)/batch_size
    loss += 1/10000*loss_function(dx, dxb)/batch_size

    try:
        zsol = odeint(net.sindy_forward, z[0, :], torch.tensor(t_train))
        loss += 0.04*loss_function(z, zsol)/batch_size
        print('pass')
    except:
        pass

    loss_unreg = np.round(loss.item(), 2)

    if epoch > 100:
        loss += lamdba*torch.norm(net.E.weight, p=1)/torch.numel(net.E.weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Coefficient mask
    mask = torch.abs(net.E.weight) > 0.2

    with torch.no_grad():
        net.E.weight *= mask

    # Update progress bar

    loss_ = np.round(loss.item(), 2)

    min_ = np.round(torch.min(net.E.weight).cpu().detach().numpy(), 2)
    n_weights = torch.sum(mask).cpu().detach().numpy()

    # min_ = np.round(torch.min(net.E.weight).detach().numpy(), 2)
    max_ = np.round(torch.max(net.E.weight).cpu().detach().numpy(), 2)

    # loop.set_description(f"Epoch [{epoch}/{epochs}]")
    # loop.set_postfix(loss=loss_, loss_unreg=loss_unreg, n_weights=n_weights, min=min_, max=max_)

    print('Epoch: {:d}/{:d} -- Loss: {:.2f} -- Loss_unreg: {:.2f} -- n_weights: {:.2f} -- min: {:.2f} -- max: {:.2f}'.format(epoch, epochs, loss_, loss_unreg, n_weights, min_, max_), flush=True)

losses_unreg.append(loss_unreg)
losses.append(loss_)

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(x.detach().numpy())
axs[0, 1].plot(dx.detach().numpy())
axs[0, 2].plot(z.detach().numpy())
axs[1, 0].plot(xb.detach().numpy())
axs[1, 1].plot(dxb.detach().numpy())
axs[1, 2].plot(zsol.detach().numpy())
fig.show()


pdb.set_trace()