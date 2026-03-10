# regime analysis: does DeepONet error spike in extreme parameter regions?
# checks error vs each Heston param to find failure modes

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deeponet import DeepONet, load_data

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def analyze():
    data = np.load('heston_data.npz')
    params = data['params']
    ivs = data['ivs']
    inputs = data['inputs']
    n_pts = int(data['n_points_per_surface'])

    n_surfaces = len(params)
    n_train = int(0.8 * n_surfaces)

    test_params = params[n_train:]
    test_ivs = ivs[n_train * n_pts:]
    n_test = len(test_params)

    # load trained model and get predictions
    (Xb_tr, Xt_tr, y_tr,
     Xb_te, Xt_te, y_te,
     stats, _, _) = load_data()

    model = DeepONet(latent=128, hidden=256).to(DEVICE)

    # retrain quickly or load — just retrain
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)

    Xb_tr, Xt_tr, y_tr = Xb_tr.to(DEVICE), Xt_tr.to(DEVICE), y_tr.to(DEVICE)
    Xb_te, Xt_te, y_te = Xb_te.to(DEVICE), Xt_te.to(DEVICE), y_te.to(DEVICE)

    print('Training model for regime analysis...')
    for epoch in range(1, 10001):
        model.train()
        idx = torch.randint(0, len(y_tr), (4096,), device=DEVICE)
        pred = model(Xb_tr[idx], Xt_tr[idx])
        loss = torch.mean((pred - y_tr[idx])**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % 2000 == 0:
            print(f'  epoch {epoch}: loss={loss.item():.7f}')

    model.eval()
    with torch.no_grad():
        pred_test = model(Xb_te, Xt_te).cpu().numpy()

    true_test = y_te.cpu().numpy()

    # per-surface errors
    surface_errors = []
    for s in range(n_test):
        sl = slice(s * n_pts, (s + 1) * n_pts)
        rel = np.mean(np.abs(pred_test[sl] - true_test[sl]) / true_test[sl]) * 100
        surface_errors.append(rel)
    surface_errors = np.array(surface_errors)

    # plot error vs each param
    param_names = ['κ (mean rev)', 'θ (long-run var)', 'σ (vol of vol)', 'ρ (correlation)', 'v₀ (init var)']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for p in range(5):
        ax = axes[p]
        ax.scatter(test_params[:, p], surface_errors, s=20, alpha=0.6)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1% target')
        ax.set_xlabel(param_names[p])
        ax.set_ylabel('Mean Relative Error (%)')
        ax.set_title(f'Error vs {param_names[p]}')
        ax.legend(); ax.grid(True, alpha=0.3)

    # Feller ratio: 2κθ/σ²
    feller = 2 * test_params[:, 0] * test_params[:, 1] / test_params[:, 2]**2
    axes[5].scatter(feller, surface_errors, s=20, alpha=0.6, color='green')
    axes[5].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1% target')
    axes[5].axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='Feller boundary')
    axes[5].set_xlabel('Feller ratio 2κθ/σ²')
    axes[5].set_ylabel('Mean Relative Error (%)')
    axes[5].set_title('Error vs Feller Condition')
    axes[5].legend(); axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regime_analysis.png', dpi=150)
    print('\nSaved: regime_analysis.png')

    # print worst surfaces
    worst = np.argsort(surface_errors)[-5:]
    print(f'\nWorst 5 test surfaces:')
    print(f'  {"Error%":>7}  {"κ":>6}  {"θ":>7}  {"σ":>6}  {"ρ":>6}  {"v0":>7}  {"Feller":>7}')
    for w in worst:
        p = test_params[w]
        f = 2 * p[0] * p[1] / p[2]**2
        print(f'  {surface_errors[w]:>7.3f}  {p[0]:>6.3f}  {p[1]:>7.4f}  {p[2]:>6.3f}  '
              f'{p[3]:>6.3f}  {p[4]:>7.4f}  {f:>7.3f}')


if __name__ == '__main__':
    analyze()
