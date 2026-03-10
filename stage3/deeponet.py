# DeepONet for Heston implied volatility surface mapping
# branch: encodes Heston params (κ, θ, σ, ρ, v0) → latent
# trunk: encodes query point (moneyness, T) → latent
# output: dot product → implied vol

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DeepONet(nn.Module):
    def __init__(self, branch_in=5, trunk_in=2, latent=128, hidden=256):
        super().__init__()
        # branch: params → latent vector
        self.branch = nn.Sequential(
            nn.Linear(branch_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent)
        )
        # trunk: (moneyness, T) → latent vector
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, query):
        b = self.branch(params)   # (batch, latent)
        t = self.trunk(query)     # (batch, latent)
        return torch.sum(b * t, dim=-1) + self.bias  # (batch,)


def load_data(path='heston_data.npz'):
    data = np.load(path)
    params = data['params']         # (n_surfaces, 5)
    inputs = data['inputs']         # (n_surfaces * 77, 2)
    ivs = data['ivs']               # (n_surfaces * 77,)
    n_pts = int(data['n_points_per_surface'])

    n_surfaces = len(params)
    n_train = int(0.8 * n_surfaces)

    # expand params to match each query point
    params_expanded = np.repeat(params, n_pts, axis=0)  # (n_surfaces * 77, 5)

    # split by surface index
    train_idx = slice(0, n_train * n_pts)
    test_idx = slice(n_train * n_pts, n_surfaces * n_pts)

    X_branch_train = torch.tensor(params_expanded[train_idx], dtype=torch.float32)
    X_trunk_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
    y_train = torch.tensor(ivs[train_idx], dtype=torch.float32)

    X_branch_test = torch.tensor(params_expanded[test_idx], dtype=torch.float32)
    X_trunk_test = torch.tensor(inputs[test_idx], dtype=torch.float32)
    y_test = torch.tensor(ivs[test_idx], dtype=torch.float32)

    # normalize inputs
    branch_mean = X_branch_train.mean(0)
    branch_std = X_branch_train.std(0) + 1e-8
    trunk_mean = X_trunk_train.mean(0)
    trunk_std = X_trunk_train.std(0) + 1e-8

    X_branch_train = (X_branch_train - branch_mean) / branch_std
    X_branch_test = (X_branch_test - branch_mean) / branch_std
    X_trunk_train = (X_trunk_train - trunk_mean) / trunk_std
    X_trunk_test = (X_trunk_test - trunk_mean) / trunk_std

    print(f'Train: {n_train} surfaces, {len(y_train)} points')
    print(f'Test:  {n_surfaces - n_train} surfaces, {len(y_test)} points')

    stats = {'branch_mean': branch_mean, 'branch_std': branch_std,
             'trunk_mean': trunk_mean, 'trunk_std': trunk_std}

    return (X_branch_train, X_trunk_train, y_train,
            X_branch_test, X_trunk_test, y_test,
            stats, n_train, n_surfaces - n_train)


def train(epochs=10000, lr=1e-3, batch_size=4096):
    (Xb_tr, Xt_tr, y_tr,
     Xb_te, Xt_te, y_te,
     stats, n_train, n_test) = load_data()

    # move to device
    Xb_tr, Xt_tr, y_tr = Xb_tr.to(DEVICE), Xt_tr.to(DEVICE), y_tr.to(DEVICE)
    Xb_te, Xt_te, y_te = Xb_te.to(DEVICE), Xt_te.to(DEVICE), y_te.to(DEVICE)

    model = DeepONet(latent=128, hidden=256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n = len(y_tr)
    history = {'train_loss': [], 'test_loss': [], 'test_rel_err': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()

        # random mini-batch
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        pred = model(Xb_tr[idx], Xt_tr[idx])
        loss = torch.mean((pred - y_tr[idx])**2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['train_loss'].append(loss.item())

        if epoch % 500 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_test = model(Xb_te, Xt_te)
                test_loss = torch.mean((pred_test - y_te)**2).item()
                rel_err = (torch.abs(pred_test - y_te) / y_te).mean().item() * 100

            history['test_loss'].append(test_loss)
            history['test_rel_err'].append(rel_err)

            print(f'  epoch {epoch:5d} | train: {loss.item():.6f} | '
                  f'test: {test_loss:.6f} | rel err: {rel_err:.3f}%')

    train_time = time.time() - t0

    # final evaluation
    model.eval()
    with torch.no_grad():
        pred_test = model(Xb_te, Xt_te)
        test_mse = torch.mean((pred_test - y_te)**2).item()
        test_mae = torch.mean(torch.abs(pred_test - y_te)).item()
        test_rel = (torch.abs(pred_test - y_te) / y_te).mean().item() * 100
        test_max_err = torch.max(torch.abs(pred_test - y_te)).item()

    print(f'\n{"="*50}')
    print(f'Final test results ({n_test} surfaces):')
    print(f'  MSE:           {test_mse:.8f}')
    print(f'  MAE:           {test_mae:.6f}')
    print(f'  Mean rel err:  {test_rel:.3f}%')
    print(f'  Max abs err:   {test_max_err:.6f}')
    print(f'  Train time:    {train_time:.1f}s')
    print(f'  Target:        <1% mean relative error')
    print(f'  {"PASS" if test_rel < 1 else "FAIL"}')

    return model, history, pred_test.cpu().numpy(), y_te.cpu().numpy(), stats, n_test


def plot_results(model, history, pred, true, n_test):
    n_pts = 77  # per surface
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # training loss
    axes[0, 0].semilogy(history['train_loss'], alpha=0.3, color='blue')
    # smooth
    window = 100
    if len(history['train_loss']) > window:
        smooth = np.convolve(history['train_loss'], np.ones(window)/window, mode='valid')
        axes[0, 0].semilogy(smooth, color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Loss'); axes[0, 0].grid(True, alpha=0.3)

    # test loss over time
    test_epochs = list(range(500, 500*len(history['test_loss'])+1, 500))
    if len(test_epochs) > len(history['test_loss']):
        test_epochs = test_epochs[:len(history['test_loss'])]
    axes[0, 1].semilogy(test_epochs, history['test_loss'], 'o-', markersize=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Test MSE')
    axes[0, 1].set_title('Test Loss'); axes[0, 1].grid(True, alpha=0.3)

    # scatter: predicted vs true IV
    axes[0, 2].scatter(true, pred, s=1, alpha=0.3)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    axes[0, 2].plot(lims, lims, 'r--', linewidth=1)
    axes[0, 2].set_xlabel('True IV'); axes[0, 2].set_ylabel('Predicted IV')
    axes[0, 2].set_title('Predicted vs True'); axes[0, 2].grid(True, alpha=0.3)

    # error distribution
    errors = pred - true
    axes[1, 0].hist(errors, bins=100, density=True, alpha=0.7)
    axes[1, 0].set_xlabel('Error (pred - true)'); axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Error Distribution (MAE={np.mean(np.abs(errors)):.5f})')
    axes[1, 0].grid(True, alpha=0.3)

    # sample surface comparison: pick first test surface
    moneyness = np.array([0.80, 0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15, 1.20])
    maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    true_surf = true[:n_pts].reshape(len(maturities), len(moneyness))
    pred_surf = pred[:n_pts].reshape(len(maturities), len(moneyness))

    # plot ATM smile (moneyness index 5 = 1.00)
    for T_idx, T_label in [(0, 'T=0.1'), (2, 'T=0.5'), (4, 'T=1.0'), (6, 'T=2.0')]:
        axes[1, 1].plot(moneyness, true_surf[T_idx, :], 'o-', label=f'True {T_label}', markersize=3)
        axes[1, 1].plot(moneyness, pred_surf[T_idx, :], 's--', label=f'Pred {T_label}', markersize=3)
    axes[1, 1].set_xlabel('Moneyness (K/S)')
    axes[1, 1].set_ylabel('Implied Vol')
    axes[1, 1].set_title('Sample Surface: True vs Predicted')
    axes[1, 1].legend(fontsize=7, ncol=2); axes[1, 1].grid(True, alpha=0.3)

    # per-surface error
    n_surfaces = len(true) // n_pts
    surface_errors = []
    for s in range(min(n_surfaces, n_test)):
        sl = slice(s * n_pts, (s+1) * n_pts)
        rel = np.mean(np.abs(pred[sl] - true[sl]) / true[sl]) * 100
        surface_errors.append(rel)

    axes[1, 2].hist(surface_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(1.0, color='red', linestyle='--', label='1% target')
    axes[1, 2].set_xlabel('Mean Relative Error (%)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'Per-Surface Error ({sum(1 for e in surface_errors if e<1)}/{len(surface_errors)} under 1%)')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('deeponet_results.png', dpi=150)
    print('Saved: deeponet_results.png')


if __name__ == '__main__':
    print(f'Device: {DEVICE}\n')
    model, history, pred, true, stats, n_test = train(epochs=10000, lr=1e-3, batch_size=4096)
    plot_results(model, history, pred, true, n_test)
