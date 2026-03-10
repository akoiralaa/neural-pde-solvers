# Deep BSDE scaling test: 2, 5, 20, 50, 100 assets
# compare against C++ MC ground truth at each dimension

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SubNet(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d)
        )
    def forward(self, x):
        return self.net(x)


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
        self.Z0 = nn.Parameter(torch.randn(d) * 0.1)
        self.subnets = nn.ModuleList([SubNet(d, hidden=max(64, d)) for _ in range(N - 1)])

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


def train_and_evaluate(d, S0, K, T, r, sigmas, corr, y0_init, epochs=6000, batch_size=4096):
    S0_t = torch.tensor(S0, dtype=torch.float32, device=DEVICE)

    # scale hidden size and time steps with dimension
    N = min(20 + d // 5, 40)
    hidden = max(64, d)
    bs = min(batch_size, 4096)  # cap batch size for memory on high-d

    model = DeepBSDE(d, T, N, r, sigmas, corr, K, y0_init).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        Y, g = model(S0_t, bs)
        loss = torch.mean((Y - g)**2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if epoch % 1000 == 0 or epoch == 1:
            print(f'  epoch {epoch:5d} | loss: {loss.item():.4f} | Y0: {model.Y0.item():.4f}')

    train_time = time.time() - t0

    # evaluate
    model.eval()
    prices = []
    with torch.no_grad():
        for _ in range(30):
            Y, _ = model(S0_t, 50000)
            prices.append(Y.mean().item())

    price = np.mean(prices)
    std = np.std(prices)

    return price, std, train_time, history


if __name__ == '__main__':
    print(f'Device: {DEVICE}\n')

    # C++ MC ground truth prices
    mc_prices = {
        2:   18.8423,
        5:   31.9565,
        20:  52.7814,
        50:  77.3326,
        100: 79.4396,
    }

    configs = [
        (2,   [0.2, 0.3],                                          15.0, 6000),
        (5,   [0.2, 0.25, 0.3, 0.22, 0.28],                       25.0, 6000),
        (20,  [0.15 + 0.01*i for i in range(20)],                  45.0, 8000),
        (50,  [0.15 + 0.005*i for i in range(50)],                 65.0, 10000),
        (100, [0.15 + 0.002*i for i in range(100)],                70.0, 12000),
    ]

    results = {}

    for d, sigmas, y0_init, epochs in configs:
        print('=' * 60)
        print(f'{d}-ASSET RAINBOW')
        print('=' * 60)

        S0 = np.array([100.0] * d)
        corr = np.full((d, d), 0.3)
        np.fill_diagonal(corr, 1.0)

        price, std, train_time, hist = train_and_evaluate(
            d, S0, 100, 1.0, 0.05, sigmas, corr, y0_init, epochs=epochs
        )

        mc_p = mc_prices[d]
        error_pct = abs(price - mc_p) / mc_p * 100

        print(f'\n  BSDE:  {price:.4f} ± {std:.4f}')
        print(f'  MC:    {mc_p:.4f}')
        print(f'  Error: {error_pct:.2f}%')
        print(f'  Train: {train_time:.1f}s\n')

        results[d] = {
            'price': price, 'std': std, 'mc': mc_p,
            'error_pct': error_pct, 'train_time': train_time, 'history': hist
        }

    # summary table
    print('\n' + '=' * 60)
    print('SCALING SUMMARY')
    print('=' * 60)
    print(f'{"Assets":>6} | {"MC Price":>10} | {"BSDE Price":>11} | {"Error":>7} | {"Train Time":>10}')
    print('-' * 60)
    for d in [2, 5, 20, 50, 100]:
        r = results[d]
        print(f'{d:>6} | {r["mc"]:>10.4f} | {r["price"]:>11.4f} | {r["error_pct"]:>6.2f}% | {r["train_time"]:>9.1f}s')

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # loss curves
    for d in [2, 5, 20, 50, 100]:
        axes[0].semilogy(results[d]['history'], alpha=0.7, label=f'{d}-asset')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss vs Dimension')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # error vs dimension
    dims = [2, 5, 20, 50, 100]
    errors = [results[d]['error_pct'] for d in dims]
    axes[1].plot(dims, errors, 'o-', linewidth=2, markersize=8)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5% target')
    axes[1].set_xlabel('Number of Assets')
    axes[1].set_ylabel('Relative Error (%)')
    axes[1].set_title('Pricing Error vs Dimension')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # training time vs dimension
    times = [results[d]['train_time'] for d in dims]
    axes[2].plot(dims, times, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Assets')
    axes[2].set_ylabel('Training Time (s)')
    axes[2].set_title('Training Time vs Dimension')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bsde_scaling.png', dpi=150)
    print('\nSaved: bsde_scaling.png')
