# fixing the 50 and 100 asset BSDE: bigger nets, more steps, residual connections

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SubNet(nn.Module):  # wider + residual connection
    def __init__(self, d, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d)
        self.skip = nn.Linear(d, d) if d != hidden else None  # residual

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out + x  # residual: subnet learns correction to identity


class DeepBSDE(nn.Module):
    def __init__(self, d, T, N, r, sigmas, corr, K, y0_init):
        super().__init__()
        self.d = d
        self.T = T
        self.N = N
        self.dt = T / N
        self.r = r
        self.K = K
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=DEVICE)
        L = np.linalg.cholesky(corr)
        self.L = torch.tensor(L, dtype=torch.float32, device=DEVICE)

        self.Y0 = nn.Parameter(torch.tensor(y0_init, dtype=torch.float32))
        self.Z0 = nn.Parameter(torch.randn(d) * 0.01)
        self.subnets = nn.ModuleList([SubNet(d, hidden=256) for _ in range(N - 1)])

    def forward(self, S0, batch_size):
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        X = S0.unsqueeze(0).expand(batch_size, -1).clone()
        Y = self.Y0 * torch.ones(batch_size, device=DEVICE)

        dW = torch.randn(batch_size, self.d, device=DEVICE) * sqrt_dt
        dW_corr = dW @ self.L.T
        Z = self.Z0.unsqueeze(0).expand(batch_size, -1)
        Y = Y - self.r * Y * dt + torch.sum(Z * self.sigmas * X * dW_corr, dim=1)
        X = X * torch.exp((self.r - 0.5*self.sigmas**2)*dt + self.sigmas*dW_corr)

        for n in range(self.N - 1):
            dW = torch.randn(batch_size, self.d, device=DEVICE) * sqrt_dt
            dW_corr = dW @ self.L.T
            Z = self.subnets[n](X / 100.0)
            Y = Y - self.r * Y * dt + torch.sum(Z * self.sigmas * X * dW_corr, dim=1)
            X = X * torch.exp((self.r - 0.5*self.sigmas**2)*dt + self.sigmas*dW_corr)

        payoff = torch.clamp(torch.max(X, dim=1).values - self.K, min=0)
        g_T = np.exp(-self.r * self.T) * payoff
        return Y, g_T


def train_and_eval(d, S0, K, T, r, sigmas, corr, y0_init, N=40, epochs=15000, batch_size=4096):
    S0_t = torch.tensor(S0, dtype=torch.float32, device=DEVICE)
    model = DeepBSDE(d, T, N, r, sigmas, corr, K, y0_init).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4000, 8000, 12000], gamma=0.3
    )

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        Y, g = model(S0_t, batch_size)
        loss = torch.mean((Y - g)**2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if epoch % 1000 == 0 or epoch == 1:
            print(f'  epoch {epoch:5d} | loss: {loss.item():.4f} | Y0: {model.Y0.item():.4f}')

    train_time = time.time() - t0

    model.eval()
    prices = []
    with torch.no_grad():
        for _ in range(50):
            Y, _ = model(S0_t, 100000)
            prices.append(Y.mean().item())

    return np.mean(prices), np.std(prices), train_time, history


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Subnets: 256-hidden, residual connections, N=40 timesteps\n')

    mc_prices = {50: 77.3326, 100: 79.4396}

    for d, y0_init in [(50, 75.0), (100, 78.0)]:
        print('=' * 60)
        print(f'{d}-ASSET RAINBOW (FIX ATTEMPT)')
        print('=' * 60)

        S0 = np.array([100.0] * d)
        if d == 50:
            sigmas = [0.15 + 0.005*i for i in range(d)]
        else:
            sigmas = [0.15 + 0.002*i for i in range(d)]
        corr = np.full((d, d), 0.3)
        np.fill_diagonal(corr, 1.0)

        price, std, t, hist = train_and_eval(
            d, S0, 100, 1.0, 0.05, sigmas, corr, y0_init,
            N=40, epochs=15000, batch_size=4096
        )

        mc_p = mc_prices[d]
        err = abs(price - mc_p) / mc_p * 100
        print(f'\n  BSDE:   {price:.4f} ± {std:.4f}')
        print(f'  MC:     {mc_p:.4f}')
        print(f'  Error:  {err:.2f}%')
        print(f'  Target: <0.50%')
        print(f'  Train:  {t:.1f}s')
        print(f'  {"PASS" if err < 0.5 else "FAIL"}\n')

    # also rerun 2-asset with better init
    print('=' * 60)
    print('2-ASSET RAINBOW (FIX Y0 INIT)')
    print('=' * 60)
    S0_2 = np.array([100.0, 100.0])
    price_2, std_2, t_2, hist_2 = train_and_eval(
        2, S0_2, 100, 1.0, 0.05, [0.2, 0.3],
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        y0_init=18.5, N=40, epochs=10000, batch_size=4096
    )
    err_2 = abs(price_2 - 18.8423) / 18.8423 * 100
    print(f'\n  BSDE:   {price_2:.4f} ± {std_2:.4f}')
    print(f'  MC:     18.8423')
    print(f'  Error:  {err_2:.2f}%')
    print(f'  {"PASS" if err_2 < 0.5 else "FAIL"}\n')
