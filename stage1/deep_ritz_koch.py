# deep ritz method for Koch snowflake eigenfunction
# minimizes Rayleigh quotient R(u) = ∫|∇u|² / ∫u² instead of PDE residual
# only needs first derivatives (no Laplacian), should be more stable

import torch
import torch.nn as nn
import numpy as np
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ---- geometry (reused from curriculum_koch.py) ----

def koch_at_level(order, side=2.0):
    h = np.sqrt(3) / 2 * side
    verts = np.array([[0.0, 2*h/3], [-side/2, -h/3], [side/2, -h/3], [0.0, 2*h/3]])
    for _ in range(order):
        new = []
        for i in range(len(verts) - 1):
            a, b = verts[i], verts[i+1]
            d = b - a
            p1 = a + d / 3; p2 = a + 2 * d / 3
            peak = (a + b) / 2 + np.array([-d[1], d[0]]) * np.sqrt(3) / 6
            new.extend([a, p1, peak, p2])
        new.append(new[0])
        verts = np.array(new)
    return verts


def star_polygon(n_pts, r_outer, r_inner):
    angles = np.linspace(0, 2*np.pi, 2*n_pts, endpoint=False) + np.pi/2
    radii = np.where(np.arange(2*n_pts) % 2 == 0, r_outer, r_inner)
    verts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([verts, verts[0]])


def circle_verts(n_pts):
    angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([verts, verts[0]])


def sample_interior(verts, n_pts, rng):
    path = Path(verts)
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)
    pts = []
    while len(pts) < n_pts:
        batch = rng.uniform([xmin, ymin], [xmax, ymax], size=(n_pts * 3, 2))
        mask = path.contains_points(batch)
        pts.extend(batch[mask])
    return np.array(pts[:n_pts])


def dist_to_boundary(xy, verts):
    v = verts[:-1]; w = verts[1:]
    ab = w - v; ab_sq = (ab**2).sum(axis=1)
    min_d = np.full(len(xy), 1e10)
    for s in range(0, len(ab), 500):
        e = min(s + 500, len(ab))
        ab_c, v_c, ab_sq_c = ab[s:e], v[s:e], ab_sq[s:e]
        ap = xy[:, None, :] - v_c[None, :, :]
        t = np.clip((ap * ab_c[None]).sum(2) / np.maximum(ab_sq_c[None], 1e-12), 0, 1)
        proj = v_c[None] + t[:, :, None] * ab_c[None]
        dists = np.sqrt(((xy[:, None] - proj)**2).sum(2))
        min_d = np.minimum(min_d, dists.min(1))
    return min_d


# ---- network ----

class FourierNet(nn.Module):
    def __init__(self, n_freqs=64, width=160, depth=5, scale=4.0):
        super().__init__()
        B = torch.randn(2, n_freqs) * scale
        self.register_buffer('B', B)
        in_dim = n_freqs * 2
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        proj = xy @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat).squeeze(-1)


class SimpleMLP(nn.Module):
    """Plain MLP without Fourier features — more stable for variational loss."""
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


# ---- Deep Ritz training ----

def sample_boundary(verts, n_pts):
    segs = np.diff(verts, axis=0)
    lengths = np.linalg.norm(segs, axis=1)
    cum = np.cumsum(lengths)
    total = cum[-1]
    s_vals = np.linspace(0, total, n_pts, endpoint=False)
    pts = []
    for s in s_vals:
        idx = min(np.searchsorted(cum, s, side='right'), len(segs) - 1)
        prev = cum[idx] - lengths[idx]
        t = (s - prev) / lengths[idx] if lengths[idx] > 1e-12 else 0.0
        pts.append(verts[idx] + t * segs[idx])
    return np.array(pts)


def train_deep_ritz(verts, n_interior=6000, n_boundary=3000, epochs=3000, lr=1e-3):
    """
    Deep Ritz with hard BC: minimize E(u) = mean(|∇u|²) - λ*mean(u²)
    with u = d(x)*NN(x) (hard BC) and learnable eigenvalue λ.
    Only needs FIRST derivatives (no Laplacian). Key advantage over PINN.
    """
    rng = np.random.default_rng(42)
    xy_np = sample_interior(verts, n_interior, rng)
    d_np = dist_to_boundary(xy_np, verts)

    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)

    model = FourierNet(n_freqs=64, width=160, depth=5, scale=4.0).to(DEVICE)

    # learnable eigenvalue
    raw_lam = nn.Parameter(torch.tensor(1.5, device=DEVICE))
    lam_init = 13.0
    raw_lam.data.fill_(np.log(np.exp(lam_init - 0.5) - 1))

    net_params = list(model.parameters())
    opt = torch.optim.Adam([
        {'params': net_params, 'lr': lr},
        {'params': [raw_lam], 'lr': lr * 5}
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'rayleigh': [], 'grad_norm': []}
    best_loss = float('inf')
    best_state = None
    best_lam = lam_init

    for ep in range(epochs):
        opt.zero_grad()
        lam = nn.functional.softplus(raw_lam) + 0.5

        raw = model(xy_t)
        u = d_t * raw  # hard BC

        # ∇u via autograd — ONLY first derivatives (no Laplacian!)
        grad_u = torch.autograd.grad(u.sum(), xy_t, create_graph=True)[0]
        grad_energy = (grad_u**2).sum(dim=1).mean()  # mean(|∇u|²)

        # energy functional: E = mean(|∇u|²) - λ*mean(u²)
        # at eigenfunction: E = 0, and λ = mean(|∇u|²)/mean(u²)
        u_sq_mean = (u**2).mean()
        energy = grad_energy - lam * u_sq_mean

        # normalization: keep u nontrivial
        norm_loss = (torch.log(u_sq_mean + 1e-8) - np.log(0.5))**2

        loss = energy**2 + 5.0 * norm_loss

        loss.backward()
        gn = nn.utils.clip_grad_norm_(net_params + [raw_lam], 10.0)
        opt.step()
        sched.step()

        history['rayleigh'].append(lam.item())
        history['grad_norm'].append(gn.item() if hasattr(gn, 'item') else gn)

        if loss.item() < best_loss and u_sq_mean.item() > 0.05:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_lam = lam.item()

        if ep % 100 == 0 or ep == epochs - 1:
            print(f'  ep {ep:5d} | E² = {energy.item()**2:.6f} | λ = {lam.item():.4f} | u² = {u_sq_mean.item():.4f}')

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, best_lam, history


# ---- PINN baseline for comparison ----

def train_pinn(verts, n_interior=6000, n_boundary=3000, epochs=2000, lr=1e-3):
    """Standard PINN: minimize |∇²u + λu|² with learnable λ."""
    rng = np.random.default_rng(42)
    xy_np = sample_interior(verts, n_interior, rng)
    d_np = dist_to_boundary(xy_np, verts)

    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)

    model = FourierNet().to(DEVICE)
    raw_lam = nn.Parameter(torch.tensor(1.5, device=DEVICE))

    # separate net params and eigenvalue param (different learning rates)
    net_params = list(model.parameters())
    opt = torch.optim.Adam([
        {'params': net_params, 'lr': lr},
        {'params': [raw_lam], 'lr': lr * 5}
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = {'phys': [], 'lam': []}
    best_loss = float('inf')
    best_state = None
    best_lam = 13.0

    # init λ near 13
    raw_lam.data.fill_(np.log(np.exp(13.0 - 0.5) - 1))

    for ep in range(epochs):
        opt.zero_grad()
        lam = nn.functional.softplus(raw_lam) + 0.5

        raw = model(xy_t)
        u = d_t * raw

        # Laplacian via autograd (second derivatives)
        grad_u = torch.autograd.grad(u.sum(), xy_t, create_graph=True)[0]
        ux, uy = grad_u[:, 0], grad_u[:, 1]
        uxx = torch.autograd.grad(ux.sum(), xy_t, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(uy.sum(), xy_t, create_graph=True)[0][:, 1]
        lap = uxx + uyy

        residual = (lap + lam * u) / (lam.detach() + 1.0)
        physics_loss = (residual**2).mean()

        u_sq = (u**2).mean()
        norm_loss = (torch.log(u_sq + 1e-8) - np.log(0.5))**2

        loss = physics_loss + 5.0 * norm_loss
        loss.backward()
        nn.utils.clip_grad_norm_(net_params + [raw_lam], 10.0)
        opt.step()
        sched.step()

        history['phys'].append(physics_loss.item())
        history['lam'].append(lam.item())

        if physics_loss.item() < best_loss and u_sq.item() > 0.05:
            best_loss = physics_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_lam = lam.item()

        if ep % 100 == 0 or ep == epochs - 1:
            print(f'  ep {ep:5d} | phys {physics_loss.item():.4f} | λ = {lam.item():.4f}')

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, best_lam, history


# ---- evaluation ----

def eval_residual(model, xy_np, d_np, lam, use_hard_bc=True):
    """Compute |∇²u + λu| on eval points."""
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    xy_t = torch.tensor(xy_np, dtype=dtype, device=dev, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=dtype, device=dev)
    model.eval()
    u = d_t * model(xy_t) if use_hard_bc else model(xy_t)
    grad_u = torch.autograd.grad(u.sum(), xy_t, create_graph=True)[0]
    ux, uy = grad_u[:, 0], grad_u[:, 1]
    uxx = torch.autograd.grad(ux.sum(), xy_t, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(uy.sum(), xy_t, create_graph=True)[0][:, 1]
    res = (uxx + uyy + lam * u).detach().cpu().numpy()
    model.train()
    return np.abs(res)


# ---- plotting ----

def plot_comparison(model_ritz, lam_ritz, hist_ritz,
                    model_pinn, lam_pinn, hist_pinn,
                    verts, xy_eval, d_eval):

    fig = plt.figure(figsize=(20, 12))

    # row 1: eigenfunctions
    tri = Delaunay(xy_eval)

    for idx, (mdl, lam_val, name, hard_bc) in enumerate([
        (model_ritz, lam_ritz, 'Deep Ritz (energy)', True),
        (model_pinn, lam_pinn, 'PINN (residual)', True)
    ]):
        mdl.eval()
        dev = next(mdl.parameters()).device
        dtype = next(mdl.parameters()).dtype
        xy_t = torch.tensor(xy_eval, dtype=dtype, device=dev)
        d_t = torch.tensor(d_eval, dtype=dtype, device=dev)
        with torch.no_grad():
            u = (d_t * mdl(xy_t)).cpu().numpy() if hard_bc else mdl(xy_t).cpu().numpy()
        if np.sum(u) < 0:
            u = -u
        u = u / (np.abs(u).max() + 1e-10)

        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.plot_trisurf(xy_eval[:, 0], xy_eval[:, 1], u,
                        triangles=tri.simplices, cmap='viridis',
                        edgecolor='none', alpha=0.95)
        ax.set_title(f'{name}\nλ ≈ {lam_val:.3f}', fontsize=13)
        ax.view_init(elev=35, azim=-55)
        ax.plot(verts[:, 0], verts[:, 1], u.min() - 0.05, 'k-', lw=0.3, alpha=0.3)

    # row 1 col 3: training curves
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.semilogy(hist_ritz['rayleigh'], color='blue', alpha=0.3, linewidth=0.5)
    w = 50
    if len(hist_ritz['rayleigh']) > w:
        smooth = np.convolve(hist_ritz['rayleigh'], np.ones(w)/w, mode='valid')
        ax3.semilogy(smooth, color='blue', linewidth=2, label='Deep Ritz R(u)')
    ax3.semilogy(hist_pinn['phys'], color='red', alpha=0.3, linewidth=0.5)
    if len(hist_pinn['phys']) > w:
        smooth_p = np.convolve(hist_pinn['phys'], np.ones(w)/w, mode='valid')
        ax3.semilogy(smooth_p, color='red', linewidth=2, label='PINN physics')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.set_title('Training Stability Comparison')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # row 2: residual heatmaps
    res_ritz = eval_residual(model_ritz, xy_eval, d_eval, lam_ritz, use_hard_bc=True)
    res_pinn = eval_residual(model_pinn, xy_eval, d_eval, lam_pinn, use_hard_bc=True)
    vmax = np.percentile(np.concatenate([res_ritz, res_pinn]), 95)

    ax4 = fig.add_subplot(2, 3, 4)
    sc1 = ax4.scatter(xy_eval[:, 0], xy_eval[:, 1], c=res_ritz,
                       s=1, cmap='hot', vmin=0, vmax=vmax)
    ax4.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.3)
    ax4.set_title(f'Deep Ritz residual\nmean = {res_ritz.mean():.2f}')
    ax4.set_aspect('equal'); plt.colorbar(sc1, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(2, 3, 5)
    sc2 = ax5.scatter(xy_eval[:, 0], xy_eval[:, 1], c=res_pinn,
                       s=1, cmap='hot', vmin=0, vmax=vmax)
    ax5.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.3)
    ax5.set_title(f'PINN residual\nmean = {res_pinn.mean():.2f}')
    ax5.set_aspect('equal'); plt.colorbar(sc2, ax=ax5, shrink=0.7)

    # row 2 col 3: λ convergence
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(hist_ritz['rayleigh'], color='blue', alpha=0.5, label=f'Deep Ritz → {lam_ritz:.2f}')
    ax6.plot(hist_pinn['lam'], color='red', alpha=0.5, label=f'PINN → {lam_pinn:.2f}')
    ax6.set_xlabel('Epoch'); ax6.set_ylabel('λ')
    ax6.set_title('Eigenvalue Convergence')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.suptitle('Koch Snowflake Eigenfunction: Deep Ritz vs PINN', fontsize=15, y=1.01)
    plt.tight_layout()

    outpath = '/Users/abhiekoirala/Desktop/neural-pde-solvers/stage1/deep_ritz_koch.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {outpath}')
    return fig


# ---- multi-domain comparison ----

def run_all_domains():
    """Run Deep Ritz on Koch, star, and circle. Compare with PINN on Koch."""
    print(f'Device: {DEVICE}\n')
    t_start = time.time()

    # --- Koch snowflake (main comparison) ---
    verts_koch = koch_at_level(4, side=2.0)
    n_pts = 8000
    n_epochs = 3000

    print('=' * 60)
    print('DEEP RITZ — Koch snowflake (level 4)')
    print('=' * 60)
    model_ritz, lam_ritz, hist_ritz = train_deep_ritz(
        verts_koch, n_interior=n_pts, epochs=n_epochs, lr=5e-4
    )
    print(f'\n  Best λ (Rayleigh) = {lam_ritz:.4f}\n')

    print('=' * 60)
    print('PINN BASELINE — Koch snowflake (level 4)')
    print('=' * 60)
    model_pinn, lam_pinn, hist_pinn = train_pinn(
        verts_koch, n_interior=n_pts, epochs=n_epochs, lr=1e-3
    )
    print(f'\n  Best λ (PINN) = {lam_pinn:.4f}\n')

    # evaluate both
    rng = np.random.default_rng(99)
    xy_eval = sample_interior(verts_koch, 10000, rng)
    d_eval = dist_to_boundary(xy_eval, verts_koch)

    res_ritz = eval_residual(model_ritz, xy_eval, d_eval, lam_ritz, use_hard_bc=True)
    res_pinn = eval_residual(model_pinn, xy_eval, d_eval, lam_pinn, use_hard_bc=True)

    print('--- Koch snowflake comparison ---')
    print(f'  Deep Ritz:  λ = {lam_ritz:.4f}, mean |∇²u + λu| = {res_ritz.mean():.4f}')
    print(f'  PINN:       λ = {lam_pinn:.4f}, mean |∇²u + λu| = {res_pinn.mean():.4f}')
    print(f'  Improvement: {res_pinn.mean() / (res_ritz.mean() + 1e-10):.1f}x')

    # stability comparison
    ritz_spikes = np.sum(np.array(hist_ritz['rayleigh']) > 10 * np.median(hist_ritz['rayleigh']))
    pinn_spikes = np.sum(np.array(hist_pinn['phys']) > 10 * np.median(hist_pinn['phys']))
    print(f'\n  Ritz loss spikes (>10x median): {ritz_spikes}')
    print(f'  PINN loss spikes (>10x median): {pinn_spikes}')

    # plot
    plot_comparison(model_ritz, lam_ritz, hist_ritz,
                    model_pinn, lam_pinn, hist_pinn,
                    verts_koch, xy_eval, d_eval)

    # --- star and circle (Deep Ritz only) ---
    domains = {
        'Five-pointed star': star_polygon(5, 1.0, 0.38),
        'Unit circle': circle_verts(200),
    }

    print('\n' + '=' * 60)
    print('DEEP RITZ — other domains')
    print('=' * 60)

    for name, verts in domains.items():
        print(f'\n  {name}:')
        _, lam, _ = train_deep_ritz(verts, n_interior=6000, epochs=1500, lr=1e-3)
        print(f'  → λ = {lam:.4f}')
        if 'circle' in name.lower():
            exact = 5.7832  # j_{0,1}² for unit disk
            print(f'    exact λ₁ = {exact:.4f}, error = {abs(lam - exact)/exact*100:.2f}%')

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed:.1f}s')


if __name__ == '__main__':
    run_all_domains()
