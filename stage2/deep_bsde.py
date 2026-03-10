# Deep BSDE solver for multi-asset rainbow option
# validated on 1D European call first (see deep_bsde_debug.py)
# payoff: (max(S1,...,Sn) - K)+

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
        self.subnets = nn.ModuleList([SubNet(d) for _ in range(N - 1)])

    def forward(self, S0, batch_size):
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        X = S0.unsqueeze(0).expand(batch_size, -1).clone()
        Y = self.Y0 * torch.ones(batch_size, device=DEVICE)

        # step 0
        dW = torch.randn(batch_size, self.d, device=DEVICE) * sqrt_dt
        dW_corr = dW @ self.L.T
        Z = self.Z0.unsqueeze(0).expand(batch_size, -1)
        Y = Y - self.r * Y * dt + torch.sum(Z * self.sigmas * X * dW_corr, dim=1)
        X = X * torch.exp((self.r - 0.5*self.sigmas**2)*dt + self.sigmas*dW_corr)

        # remaining steps
        for n in range(self.N - 1):
            dW = torch.randn(batch_size, self.d, device=DEVICE) * sqrt_dt
            dW_corr = dW @ self.L.T
            Z = self.subnets[n](X / 100.0)  # normalize
            Y = Y - self.r * Y * dt + torch.sum(Z * self.sigmas * X * dW_corr, dim=1)
            X = X * torch.exp((self.r - 0.5*self.sigmas**2)*dt + self.sigmas*dW_corr)

        payoff = torch.clamp(torch.max(X, dim=1).values - self.K, min=0)
        g_T = np.exp(-self.r * self.T) * payoff
        return Y, g_T


def train(d, S0, K, T, r, sigmas, corr, y0_init,
          N=20, epochs=6000, batch_size=4096, lr=5e-3):
    S0_t = torch.tensor(S0, dtype=torch.float32, device=DEVICE)
    model = DeepBSDE(d, T, N, r, sigmas, corr, K, y0_init).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        Y, g = model(S0_t, batch_size)
        loss = torch.mean((Y - g)**2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if epoch % 500 == 0 or epoch == 1:
            print(f'epoch {epoch:5d} | loss: {loss.item():.4f} | Y0: {model.Y0.item():.4f}')

    return model, history


def evaluate(model, S0, batch_size=100000, n_runs=30):
    S0_t = torch.tensor(S0, dtype=torch.float32, device=DEVICE)
    model.eval()
    prices = []
    with torch.no_grad():
        for _ in range(n_runs):
            Y, _ = model(S0_t, batch_size)
            prices.append(Y.mean().item())
    return np.mean(prices), np.std(prices)


def greeks(model, S0, bump=1.0):
    d = len(S0)
    p0, _ = evaluate(model, S0)
    deltas, gammas = np.zeros(d), np.zeros(d)
    for i in range(d):
        Su, Sd = S0.copy(), S0.copy()
        Su[i] += bump; Sd[i] -= bump
        pu, _ = evaluate(model, Su)
        pd, _ = evaluate(model, Sd)
        deltas[i] = (pu - pd) / (2*bump)
        gammas[i] = (pu - 2*p0 + pd) / bump**2
    return p0, deltas, gammas


if __name__ == '__main__':
    # 2-asset — init Y0 near MC price
    print('=' * 50)
    print('2-ASSET DEEP BSDE')
    print('=' * 50)
    S0_2 = np.array([100.0, 100.0])
    sig2 = [0.2, 0.3]
    corr2 = np.array([[1.0, 0.5], [0.5, 1.0]])

    t0 = time.time()
    m2, h2 = train(2, S0_2, 100, 1.0, 0.05, sig2, corr2, y0_init=15.0)
    t2 = time.time() - t0

    p2, s2 = evaluate(m2, S0_2)
    print(f'\nBSDE:  {p2:.4f} ± {s2:.4f}')
    print(f'MC:    18.8245')
    print(f'FDM:   18.7773')
    print(f'Error: {abs(p2-18.8245)/18.8245*100:.2f}%')
    print(f'Time:  {t2:.1f}s')

    p2g, d2, g2 = greeks(m2, S0_2)
    print('Greeks:')
    for i in range(2):
        print(f'  Asset {i+1}: Delta={d2[i]:.4f} Gamma={g2[i]:.6f}')

    # 5-asset
    print('\n' + '=' * 50)
    print('5-ASSET DEEP BSDE')
    print('=' * 50)
    S0_5 = np.array([100.0]*5)
    sig5 = [0.2, 0.25, 0.3, 0.22, 0.28]
    corr5 = np.full((5,5), 0.3); np.fill_diagonal(corr5, 1.0)

    t0 = time.time()
    m5, h5 = train(5, S0_5, 100, 1.0, 0.05, sig5, corr5, y0_init=25.0, epochs=8000)
    t5 = time.time() - t0

    p5, s5 = evaluate(m5, S0_5)
    print(f'\nBSDE:  {p5:.4f} ± {s5:.4f}')
    print(f'MC:    31.9441')
    print(f'Error: {abs(p5-31.9441)/31.9441*100:.2f}%')
    print(f'Time:  {t5:.1f}s')

    p5g, d5, g5 = greeks(m5, S0_5)
    print('Greeks:')
    for i in range(5):
        print(f'  Asset {i+1} (σ={sig5[i]}): Delta={d5[i]:.4f} Gamma={g5[i]:.6f}')

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogy(h2, alpha=0.6, label='2-asset')
    axes[0].semilogy(h5, alpha=0.6, label='5-asset')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Deep BSDE Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    mc_p = [18.8245, 31.9441]; bsde_p = [p2, p5]
    x = np.arange(2)
    axes[1].bar(x-0.15, mc_p, 0.3, label='MC (10M)', color='steelblue')
    axes[1].bar(x+0.15, bsde_p, 0.3, label='Deep BSDE', color='coral')
    axes[1].set_xticks(x); axes[1].set_xticklabels(['2-Asset', '5-Asset'])
    axes[1].set_ylabel('Price'); axes[1].set_title('Price: MC vs Deep BSDE')
    axes[1].legend()

    mc_d5 = [0.1866, 0.2376, 0.2892, 0.2067, 0.2686]
    x5 = np.arange(5)
    axes[2].bar(x5-0.15, mc_d5, 0.3, label='MC', color='steelblue')
    axes[2].bar(x5+0.15, d5, 0.3, label='Deep BSDE', color='coral')
    axes[2].set_xticks(x5); axes[2].set_xticklabels([f'S{i+1}' for i in range(5)])
    axes[2].set_ylabel('Delta'); axes[2].set_title('Delta: MC vs Deep BSDE (5-Asset)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('deep_bsde_results.png', dpi=150)
    print('\nSaved: deep_bsde_results.png')

    torch.save(m2.state_dict(), 'model_bsde_2asset.pt')
    torch.save(m5.state_dict(), 'model_bsde_5asset.pt')
    print('Models saved.')
