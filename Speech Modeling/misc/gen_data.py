import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def nonlinear_transition_ar(z_prev, coefs, nonlinearity, noise_std):
    # AR process: z_t = sum(coefs[i] * z_{t-i}) for i in [1, order]
    z_next = np.sum([coefs[i] * z_prev[-(i+1)] for i in range(len(coefs))], axis=0)
    z_next = 0.5 * z_next + 0.5 * nonlinearity(z_next)
    z_next += np.random.normal(0, noise_std, z_next.shape)
    return z_next

def nonlinear_transition(z_prev, A, nonlinearity, noise_std):
    z_linear = np.dot(A, z_prev)
    z_nonlinear = 0.5 * z_linear + 0.5 * nonlinearity(z_linear)
    z_next = z_nonlinear + np.random.normal(0, noise_std, z_prev.shape)
    return z_next

# Modify the generate_system function to use an AR process for Z0
def generate_system_ar(T, coefs, Ax, Ay, Wx, Wy, initial_states, noise_std=0.1, epsilon_x=0.05, epsilon_y=0.05):
    order = len(coefs)  # Autoregressive order
    Z0 = np.zeros((T + order,))  # Include space for initial conditions
    Zx = np.zeros((T + 1, Ax.shape[0]))
    Zy = np.zeros((T + 1, Ay.shape[0]))
    X = np.zeros((T, Wx.shape[0]))
    Y = np.zeros((T, Wy.shape[0]))

    # Set initial conditions for Z0, Zx, Zy
    Z0[:order] = initial_states['z0']
    Zx[0, :], Zy[0, :] = initial_states['zx'], initial_states['zy']

    for t in range(order, T + order):
        Z0[t] = nonlinear_transition_ar(Z0[(t-order):t], coefs, np.sin, noise_std)
        Zx[t-order+1, :] = nonlinear_transition(Zx[t-order, :], Ax, np.tanh, noise_std)
        Zy[t-order+1, :] = nonlinear_transition(Zy[t-order, :], Ay, np.tanh, noise_std)

        x_state = np.concatenate(([Z0[t]], Zx[t-order+1, :]))
        y_state = np.concatenate(([Z0[t]], Zy[t-order+1, :]))
        X[t-order, :] = np.dot(Wx, x_state) + np.random.normal(0, epsilon_x, Wx.shape[0])
        Y[t-order, :] = np.dot(Wy, y_state) + np.random.normal(0, epsilon_y, Wy.shape[0])

    return Z0[order:], Zx[1:, :], Zy[1:, :], X, Y