# fixing the 50 and 100 asset BSDE: bigger nets, more steps, residual connections

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SubNet(nn.Module):
    def __init__(self, d, hidden=256, use_residual=True):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d)
        self.use_residual = use_residual

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out + x if self.use_residual else out


class DeepBSDE(nn.Module):
    def __init__(self, d, T, N, r, sigmas, corr, K, y0_init, hidden=256, use_residual=True):
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
        self.subnets = nn.ModuleList([SubNet(d, hidden=hidden, use_residual=use_residual) for _ in range(N - 1)])

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


def train_and_eval(d, S0, K, T, r, sigmas, corr, y0_init, N=40, epochs=15000, batch_size=4096, hidden=256, use_residual=True):
    S0_t = torch.tensor(S0, dtype=torch.float32, device=DEVICE)
    model = DeepBSDE(d, T, N, r, sigmas, corr, K, y0_init, hidden=hidden, use_residual=use_residual).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[4000, 8000, 12000], gamma=0.3
    )

    history = []
    y0_history = []
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
        y0_history.append(model.Y0.item())
        if epoch % 1000 == 0 or epoch == 1:
            print(f'  epoch {epoch:5d} | loss: {loss.item():.4f} | Y0: {model.Y0.item():.4f}')

    train_time = time.time() - t0

    model.eval()
    prices = []
    with torch.no_grad():
        for _ in range(50):
            Y, _ = model(S0_t, 100000)
            prices.append(Y.mean().item())

    return np.mean(prices), np.std(prices), train_time, history, y0_history


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Subnets: 256-hidden, residual connections, N=40 timesteps\n')

    mc_prices = {2: 18.8423, 5: 31.9565, 20: 52.7814, 50: 77.3326, 100: 79.4396}

    # hidden=64 no residual for low-d, 256+residual for high-d
    # residual connection hurts at d=2 (biases Z toward identity, 5.35% error)
    configs = [
        (2,   [0.2, 0.3],                              18.5, 10000, 64,  False),
        (5,   [0.2, 0.25, 0.3, 0.22, 0.28],            25.0, 10000, 64,  False),
        (20,  [0.15 + 0.01*i for i in range(20)],       45.0, 12000, 256, True),
        (50,  [0.15 + 0.005*i for i in range(50)],      75.0, 15000, 256, True),
        (100, [0.15 + 0.002*i for i in range(100)],     78.0, 15000, 256, True),
    ]

    results = {}

    for d, sigmas, y0_init, epochs, hidden, residual in configs:
        print('=' * 60)
        print(f'{d}-ASSET RAINBOW (hidden={hidden}, {"residual" if residual else "no residual"})')
        print('=' * 60)

        S0 = np.array([100.0] * d)
        if d == 2:
            corr = np.array([[1.0, 0.5], [0.5, 1.0]])  # match MC reference (rho=0.5)
        else:
            corr = np.full((d, d), 0.3)
            np.fill_diagonal(corr, 1.0)

        price, std, t, hist, y0_hist = train_and_eval(
            d, S0, 100, 1.0, 0.05, sigmas, corr, y0_init,
            N=40, epochs=epochs, batch_size=4096, hidden=hidden, use_residual=residual
        )

        mc_p = mc_prices[d]
        err = abs(price - mc_p) / mc_p * 100
        print(f'\n  BSDE:   {price:.4f} ± {std:.4f}')
        print(f'  MC:     {mc_p:.4f}')
        print(f'  Error:  {err:.2f}%')
        print(f'  Target: <0.50%')
        print(f'  Train:  {t:.1f}s')
        print(f'  {"PASS" if err < 0.5 else "FAIL"}\n')

        results[d] = {
            'price': price, 'std': std, 'mc': mc_p,
            'error_pct': err, 'train_time': t,
            'history': hist, 'y0_history': y0_hist
        }

    # summary table
    print('\n' + '=' * 60)
    print('SCALING SUMMARY (FIXED)')
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
    axes[0].set_title('Training Loss')
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

    # Y0 convergence
    for d in [2, 5, 20, 50, 100]:
        mc_p = results[d]['mc']
        y0_norm = [y / mc_p for y in results[d]['y0_history']]  # normalize by target
        axes[2].plot(y0_norm, alpha=0.7, label=f'{d}-asset')
    axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Y0 / MC Price')
    axes[2].set_title('Y0 Convergence (normalized)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bsde_scaling_fixed.png', dpi=150)
    print('\nSaved: bsde_scaling_fixed.png')
