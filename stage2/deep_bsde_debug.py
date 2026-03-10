# debug script: verify BSDE machinery on 1D European call first
# if this doesn't converge, the architecture is wrong
# if it does, the rainbow payoff is the problem

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def bs_call_price(S0, K, T, r, sigma):  # Black-Scholes analytical
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


class SubNet(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)


class DeepBSDE_1D(nn.Module):
    def __init__(self, T, N, r, sigma, K, S0):
        super().__init__()
        self.T = T
        self.N = N
        self.dt = T / N
        self.r = r
        self.sigma = sigma
        self.K = K

        # Y0 and Z0 are free parameters
        true_price = bs_call_price(S0, K, T, r, sigma)
        self.Y0 = nn.Parameter(torch.tensor(true_price * 0.5, dtype=torch.float32))  # init at half true price
        self.Z0 = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.subnets = nn.ModuleList([SubNet(1, 1) for _ in range(N - 1)])

    def forward(self, S0_val, batch_size):
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        X = S0_val * torch.ones(batch_size, 1, device=DEVICE)
        Y = self.Y0 * torch.ones(batch_size, device=DEVICE)

        # step 0
        dW = torch.randn(batch_size, 1, device=DEVICE) * sqrt_dt
        Z = self.Z0 * torch.ones(batch_size, 1, device=DEVICE)
        Y = Y - self.r * Y * dt + (Z * self.sigma * X * dW).squeeze()
        X = X * torch.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*dW)

        for n in range(self.N - 1):
            dW = torch.randn(batch_size, 1, device=DEVICE) * sqrt_dt
            Z = self.subnets[n](X / S0_val)  # normalize input
            Y = Y - self.r * Y * dt + (Z * self.sigma * X * dW).squeeze()
            X = X * torch.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*dW)

        payoff = torch.clamp(X.squeeze() - self.K, min=0)
        g_T = np.exp(-self.r * self.T) * payoff
        return Y, g_T


if __name__ == '__main__':
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    true_price = bs_call_price(S0, K, T, r, sigma)
    print(f'BS analytical price: {true_price:.4f}')
    print(f'Device: {DEVICE}\n')

    model = DeepBSDE_1D(T, N=20, r=r, sigma=sigma, K=K, S0=S0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    t0 = time.time()
    history = []
    for epoch in range(1, 4001):
        model.train()
        Y, g = model(S0, batch_size=4096)
        loss = torch.mean((Y - g)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if epoch % 200 == 0 or epoch == 1:
            print(f'epoch {epoch:4d} | loss: {loss.item():.4f} | Y0: {model.Y0.item():.4f} | true: {true_price:.4f}')

    elapsed = time.time() - t0

    # evaluate
    model.eval()
    prices = []
    with torch.no_grad():
        for _ in range(50):
            Y, _ = model(S0, 100000)
            prices.append(Y.mean().item())
    pred = np.mean(prices)
    std = np.std(prices)

    print(f'\nResult: {pred:.4f} ± {std:.4f}')
    print(f'True:   {true_price:.4f}')
    print(f'Error:  {abs(pred - true_price)/true_price*100:.2f}%')
    print(f'Time:   {elapsed:.1f}s')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history)
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='target loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'1D European Call — Deep BSDE Debug (true={true_price:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bsde_debug.png', dpi=150)
    print('Saved: bsde_debug.png')
