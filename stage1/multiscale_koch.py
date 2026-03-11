# adaptive multi-scale PINN for Koch snowflake eigenfunction
# strategy: train base solution, compute residual map, adaptively
# resample collocation points where residual is highest, retrain
# repeat at each fractal refinement level

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time
from experiment_log import ExperimentLog

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ---- Koch geometry ----

def koch_at_level(order, side=2.0):
    h = np.sqrt(3) / 2 * side
    verts = np.array([
        [0.0, 2*h/3], [-side/2, -h/3], [side/2, -h/3], [0.0, 2*h/3]
    ])
    for _ in range(order):
        new = []
        for i in range(len(verts) - 1):
            a, b = verts[i], verts[i+1]
            d = b - a
            p1 = a + d / 3
            p2 = a + 2 * d / 3
            peak = (a + b) / 2 + np.array([-d[1], d[0]]) * np.sqrt(3) / 6
            new.extend([a, p1, peak, p2])
        new.append(new[0])
        verts = np.array(new)
    return verts


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
    v = verts[:-1]
    w = verts[1:]
    ab = w - v
    ab_sq = (ab**2).sum(axis=1)
    min_d = np.full(len(xy), 1e10)
    chunk = 500
    for start in range(0, len(ab), chunk):
        end = min(start + chunk, len(ab))
        ab_c, v_c, ab_sq_c = ab[start:end], v[start:end], ab_sq[start:end]
        ap = xy[:, None, :] - v_c[None, :, :]
        t = np.clip((ap * ab_c[None, :, :]).sum(axis=2) / np.maximum(ab_sq_c[None, :], 1e-12), 0, 1)
        proj = v_c[None, :, :] + t[:, :, None] * ab_c[None, :, :]
        dists = np.sqrt(((xy[:, None, :] - proj)**2).sum(axis=2))
        min_d = np.minimum(min_d, dists.min(axis=1))
    return min_d


# ---- network ----

class FourierNet(nn.Module):
    def __init__(self, n_freqs=48, width=128, depth=5, scale=3.0):
        super().__init__()
        B = torch.randn(2, n_freqs) * scale
        self.register_buffer('B', B)
        in_dim = n_freqs * 2
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.raw_lam = nn.Parameter(torch.tensor(1.5))

    @property
    def lam(self):
        return nn.functional.softplus(self.raw_lam) + 0.5

    def forward(self, xy):
        proj = xy @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat).squeeze(-1)


def compute_laplacian(u, xy):
    grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    ux, uy = grad_u[:, 0], grad_u[:, 1]
    uxx = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:, 1]
    return uxx + uyy


def compute_residual(model, xy_np, d_np, verts):
    """Compute |∇²u + λu| at given points."""
    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
    model.eval()
    u = d_t * model(xy_t)
    lap = compute_laplacian(u, xy_t)
    lam = model.lam
    res = (lap + lam * u).detach().cpu().numpy()
    return np.abs(res)


def compute_importance_weights(model, xy_t, d_t, n_pts):
    """Compute per-point importance weights from PDE residual. No points are replaced."""
    model.eval()
    # need autograd for Laplacian — can't use torch.no_grad()
    xy_tmp = xy_t.detach().clone().requires_grad_(True)
    raw = model(xy_tmp)
    u = d_t * raw
    grad_u = torch.autograd.grad(u.sum(), xy_tmp, create_graph=True)[0]
    ux, uy = grad_u[:, 0], grad_u[:, 1]
    uxx = torch.autograd.grad(ux.sum(), xy_tmp, retain_graph=True, create_graph=False)[0][:, 0]
    uyy = torch.autograd.grad(uy.sum(), xy_tmp, create_graph=False)[0][:, 1]
    lap = uxx + uyy
    lam = model.lam.detach()
    res = torch.abs(lap + lam * u).detach()

    # weights = softmax of log-residual (avoids extreme ratios)
    # clamp to prevent any single point from dominating
    log_res = torch.log(res + 1e-8)
    weights = torch.softmax(log_res, dim=0) * n_pts  # mean weight = 1
    weights = torch.clamp(weights, 0.1, 10.0)  # no point gets >10x or <0.1x

    model.train()
    return weights.detach()


def adaptive_sample(model, verts, n_uniform, n_adaptive, rng):
    """Sample points adaptively: uniform + concentrated where residual is high."""
    # uniform sample
    xy_uniform = sample_interior(verts, n_uniform, rng)
    d_uniform = dist_to_boundary(xy_uniform, verts)

    # compute residual on a probe grid
    xy_probe = sample_interior(verts, n_uniform * 2, rng)
    d_probe = dist_to_boundary(xy_probe, verts)
    res = compute_residual(model, xy_probe, d_probe, verts)

    # sample adaptively: probability proportional to residual
    weights = res**2
    weights /= weights.sum()
    idx = rng.choice(len(xy_probe), size=n_adaptive, p=weights, replace=True)
    # jitter slightly to avoid exact duplicates
    xy_adaptive = xy_probe[idx] + rng.normal(0, 0.005, size=(n_adaptive, 2))

    # filter to keep only interior points
    path = Path(verts)
    inside = path.contains_points(xy_adaptive)
    xy_adaptive = xy_adaptive[inside]
    d_adaptive = dist_to_boundary(xy_adaptive, verts)

    xy_all = np.vstack([xy_uniform, xy_adaptive])
    d_all = np.concatenate([d_uniform, d_adaptive])
    return xy_all, d_all


def train_importance_weighted(verts, lam_init=13.0, epochs=20000, n_pts=8000,
                              n_boundary=3000, lr=1e-3, reweight_every=2000):
    """
    Importance-weighted PINN: fixed collocation points, adaptive per-point weights.
    Points never change — only their weights in the loss function update periodically.
    This eliminates the distribution shift that causes training spikes.
    """
    rng = np.random.default_rng(42)

    model = FourierNet(n_freqs=64, width=160, depth=5, scale=4.0).to(DEVICE)

    # init lambda
    target_sp = lam_init - 0.5
    model.raw_lam.data.fill_(np.log(np.exp(target_sp) - 1) if target_sp > 0 else 1.0)

    # fixed collocation points — sampled once, never replaced
    xy_np = sample_interior(verts, n_pts, rng)
    d_np = dist_to_boundary(xy_np, verts)
    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
    xy_bdy = torch.tensor(sample_boundary(verts, n_boundary), dtype=torch.float32, device=DEVICE)

    # start with uniform weights
    weights = torch.ones(n_pts, device=DEVICE)

    opt = torch.optim.Adam(
        [{'params': [p for n, p in model.named_parameters() if n != 'raw_lam'], 'lr': lr},
         {'params': [model.raw_lam], 'lr': lr * 5}]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss, best_state, best_lam = float('inf'), None, lam_init
    history = {'loss': [], 'phys': [], 'lam': [], 'reweight_epochs': []}

    for ep in range(epochs):
        # periodically recompute importance weights (but NEVER replace points)
        if ep > 0 and ep % reweight_every == 0:
            weights = compute_importance_weights(model, xy_t, d_t, n_pts)
            history['reweight_epochs'].append(ep)

        opt.zero_grad()
        lam = model.lam

        raw = model(xy_t)
        u = d_t * raw
        lap = compute_laplacian(u, xy_t)

        residual = (lap + lam * u) / (lam.detach() + 1.0)
        # weighted mean: high-residual points count more
        physics_loss = (weights * residual**2).mean()

        u_sq = (u**2).mean()
        norm_loss = (torch.log(u_sq + 1e-8) - np.log(0.5))**2

        u_bdy = model(xy_bdy)
        bdy_loss = (u_bdy**2).mean()

        loss = physics_loss + 5.0 * norm_loss + 0.5 * bdy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        history['loss'].append(loss.item())
        history['phys'].append(physics_loss.item())
        history['lam'].append(lam.item())

        with torch.no_grad():
            if loss.item() < best_loss and u_sq.item() > 0.05:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_lam = lam.item()

        if ep % 1000 == 0 or ep == epochs - 1:
            w_min, w_max = weights.min().item(), weights.max().item()
            print(f'  ep {ep:5d} | loss {loss.item():.5f} | '
                  f'phys {physics_loss.item():.6f} | u² {u_sq.item():.4f} | '
                  f'λ {lam.item():.4f} | w:[{w_min:.2f},{w_max:.2f}]')

    if best_state:
        model.load_state_dict(best_state)
    print(f'\nFinal λ = {best_lam:.4f}')
    return model, best_lam, history


def train_adaptive(verts, lam_init=13.0, n_phases=4, epochs_per_phase=5000,
                   n_uniform=5000, n_adaptive=3000, n_boundary=3000, lr=1e-3):
    """Train with adaptive resampling: each phase recomputes residual and focuses points there."""
    rng = np.random.default_rng(42)

    model = FourierNet(n_freqs=64, width=160, depth=5, scale=4.0).to(DEVICE)

    # init lambda
    target_sp = lam_init - 0.5
    model.raw_lam.data.fill_(np.log(np.exp(target_sp) - 1) if target_sp > 0 else 1.0)

    xy_bdy = torch.tensor(sample_boundary(verts, n_boundary), dtype=torch.float32, device=DEVICE)

    total_epochs = 0
    best_loss, best_state, best_lam = float('inf'), None, lam_init
    history = {'loss': [], 'phys': [], 'lam': [], 'phase_starts': []}

    for phase in range(n_phases):
        print(f'\n--- Phase {phase} ---')
        history['phase_starts'].append(total_epochs)

        if phase == 0:
            # uniform sampling for first phase
            xy_np = sample_interior(verts, n_uniform + n_adaptive, rng)
            d_np = dist_to_boundary(xy_np, verts)
        else:
            # adaptive resampling based on residual from previous phase
            xy_np, d_np = adaptive_sample(model, verts, n_uniform, n_adaptive, rng)
            n_actual = len(xy_np)
            print(f'  Adaptive points: {n_actual} ({n_uniform} uniform + '
                  f'{n_actual - n_uniform} from residual)')

        d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
        xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)

        # lr schedule: warm restart each phase but decaying base lr
        phase_lr = lr * (0.5 ** phase)
        opt = torch.optim.Adam(
            [{'params': [p for n, p in model.named_parameters() if n != 'raw_lam'], 'lr': phase_lr},
             {'params': [model.raw_lam], 'lr': phase_lr * 5}]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_per_phase, eta_min=1e-6)

        for ep in range(epochs_per_phase):
            # resample every 2000 epochs within phase (only if using adaptive points)
            if n_adaptive > 0 and ep > 0 and ep % 2000 == 0:
                if phase > 0:
                    xy_np, d_np = adaptive_sample(model, verts, n_uniform, n_adaptive, rng)
                else:
                    xy_np = sample_interior(verts, n_uniform + n_adaptive, rng)
                    d_np = dist_to_boundary(xy_np, verts)
                d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
                xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)

            opt.zero_grad()
            lam = model.lam

            raw = model(xy_t)
            u = d_t * raw
            lap = compute_laplacian(u, xy_t)

            residual = (lap + lam * u) / (lam.detach() + 1.0)
            physics_loss = (residual**2).mean()

            u_sq = (u**2).mean()
            norm_loss = (torch.log(u_sq + 1e-8) - np.log(0.5))**2

            u_bdy = model(xy_bdy)
            bdy_loss = (u_bdy**2).mean()

            loss = physics_loss + 5.0 * norm_loss + 0.5 * bdy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()

            history['loss'].append(loss.item())
            history['phys'].append(physics_loss.item())
            history['lam'].append(lam.item())

            with torch.no_grad():
                if loss.item() < best_loss and u_sq.item() > 0.05:
                    best_loss = loss.item()
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_lam = lam.item()

            if ep % 1000 == 0 or ep == epochs_per_phase - 1:
                print(f'  ep {total_epochs:5d} | loss {loss.item():.5f} | '
                      f'phys {physics_loss.item():.6f} | u² {u_sq.item():.4f} | λ {lam.item():.4f}')

            total_epochs += 1

    if best_state:
        model.load_state_dict(best_state)
    print(f'\nFinal λ = {best_lam:.4f}')
    return model, best_lam, history


def plot_three_way(models, lams, histories, labels, verts, outpath):
    """Compare three methods: uniform, point-swap adaptive, importance-weighted."""
    path = Path(verts)
    xmin, ymin = verts.min(axis=0) - 0.03
    xmax, ymax = verts.max(axis=0) + 0.03
    res = 280

    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xi, yi)
    flat = np.column_stack([X.ravel(), Y.ravel()])
    inside = path.contains_points(flat)
    xy_in = flat[inside]
    d_in = dist_to_boundary(xy_in, verts)

    xy_t = torch.tensor(xy_in, dtype=torch.float32, device=DEVICE)
    d_t = torch.tensor(d_in, dtype=torch.float32, device=DEVICE)

    # triangulation for plotting
    tri = mtri.Triangulation(xy_in[:, 0], xy_in[:, 1])
    max_edge = (xmax - xmin) / res * 3.5
    ts = tri.triangles
    xt, yt = xy_in[ts, 0], xy_in[ts, 1]
    tri.set_mask((xt.max(1) - xt.min(1) > max_edge) | (yt.max(1) - yt.min(1) > max_edge))

    # compute solutions and residuals for each model
    results = {}
    for name, mdl in zip(labels, models):
        mdl.eval()
        with torch.no_grad():
            u = (d_t * mdl(xy_t)).cpu().numpy()
        if np.sum(u) < 0:
            u = -u
        u_norm = u / (np.abs(u).max() + 1e-10)
        res_vals = compute_residual(mdl, xy_in, d_in, verts)
        results[name] = {'u': u_norm, 'residual': res_vals}

    fig = plt.figure(figsize=(21, 14))

    # row 1: eigenfunctions (3 columns)
    for i, (name, lam_val) in enumerate(zip(labels, lams)):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.plot_trisurf(tri, results[name]['u'], cmap='viridis', edgecolor='none', alpha=0.95)
        ax.set_title(f'{name}\n(λ ≈ {lam_val:.3f})', fontsize=12)
        ax.view_init(elev=35, azim=-55)
        z_floor = float(results[name]['u'].min()) - 0.05
        ax.plot(verts[:, 0], verts[:, 1], z_floor, 'k-', linewidth=0.3, alpha=0.3)

    # row 2 left: residual heatmaps side by side
    vmax = np.percentile(np.concatenate([results[n]['residual'] for n in labels]), 95)

    ax4 = fig.add_subplot(234)
    sc = ax4.scatter(xy_in[:, 0], xy_in[:, 1], c=results[labels[1]]['residual'],
                     s=1, cmap='hot', vmin=0, vmax=vmax)
    ax4.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.5)
    ax4.set_title(f'{labels[1]} residual\nmean = {results[labels[1]]["residual"].mean():.4f}')
    ax4.set_aspect('equal'); plt.colorbar(sc, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(235)
    sc = ax5.scatter(xy_in[:, 0], xy_in[:, 1], c=results[labels[2]]['residual'],
                     s=1, cmap='hot', vmin=0, vmax=vmax)
    ax5.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.5)
    ax5.set_title(f'{labels[2]} residual\nmean = {results[labels[2]]["residual"].mean():.4f}')
    ax5.set_aspect('equal'); plt.colorbar(sc, ax=ax5, shrink=0.7)

    # row 2 right: training curves overlay
    ax6 = fig.add_subplot(236)
    colors = ['gray', 'blue', 'green']
    w = 200
    for hist, label, color in zip(histories, labels, colors):
        phys = hist['phys']
        ax6.semilogy(phys, alpha=0.15, color=color, linewidth=0.5)
        if len(phys) > w:
            smooth = np.convolve(phys, np.ones(w)/w, mode='valid')
            ax6.semilogy(smooth, color=color, linewidth=2, label=label)
        # mark resampling events
        if 'phase_starts' in hist:
            for ps in hist['phase_starts'][1:]:
                ax6.axvline(ps, color=color, linestyle='--', alpha=0.3)
        if 'reweight_epochs' in hist:
            for rw in hist['reweight_epochs']:
                ax6.axvline(rw, color=color, linestyle=':', alpha=0.2)
    ax6.set_xlabel('Epoch'); ax6.set_ylabel('Physics Loss')
    ax6.set_title('Training Stability')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    # count catastrophic spikes: loss > 100 in second half of training
    spike_counts = []
    for hist, label in zip(histories, labels):
        phys = np.array(hist['phys'])
        half = len(phys) // 2
        spikes = int(np.sum(phys[half:] > 100.0))
        spike_counts.append(spikes)

    # stats text
    mean_residuals = [results[n]['residual'].mean() for n in labels]
    stats = '  |  '.join([f'{n}: {r:.4f} ({s} spikes)' for n, r, s
                          in zip(labels, mean_residuals, spike_counts)])

    fig.text(0.5, 0.02, f'Mean residual — {stats}',
             ha='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Koch Snowflake Eigenfunction — Adaptive Strategies Compared', fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {outpath}')
    return fig


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    t_start = time.time()

    # Koch snowflake at level 4 (finest)
    verts = koch_at_level(4, side=2.0)
    print(f'Koch boundary: {len(verts)-1} edges')

    N_EPOCHS = 10000

    # 1) uniform baseline (fixed points, uniform weights)
    print('\n' + '='*60)
    print('1. UNIFORM BASELINE (fixed points, no adaptation)')
    print('='*60)
    model_uniform, lam_uniform, hist_uniform = train_adaptive(
        verts, lam_init=13.0,
        n_phases=1, epochs_per_phase=N_EPOCHS,
        n_uniform=8000, n_adaptive=0, n_boundary=3000, lr=1e-3
    )

    # 2) old adaptive (point swapping — causes spikes)
    print('\n' + '='*60)
    print('2. POINT-SWAP ADAPTIVE (old method — spikes expected)')
    print('='*60)
    n_phases_swap = max(2, N_EPOCHS // 5000)
    model_swap, lam_swap, hist_swap = train_adaptive(
        verts, lam_init=13.0,
        n_phases=n_phases_swap, epochs_per_phase=N_EPOCHS // n_phases_swap,
        n_uniform=4000, n_adaptive=4000, n_boundary=3000, lr=1e-3
    )

    # 3) importance-weighted (fixed points, adaptive weights — no spikes)
    print('\n' + '='*60)
    print('3. IMPORTANCE-WEIGHTED (fixed points, adaptive weights)')
    print('='*60)
    model_iw, lam_iw, hist_iw = train_importance_weighted(
        verts, lam_init=13.0,
        epochs=N_EPOCHS, n_pts=8000, n_boundary=3000, lr=1e-3,
        reweight_every=2000
    )

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed:.1f}s')

    # residual comparison
    print('\n--- Residual comparison ---')
    xy_eval = sample_interior(verts, 10000, np.random.default_rng(99))
    d_eval = dist_to_boundary(xy_eval, verts)

    # experiment logging
    log = ExperimentLog("stage1/multiscale_koch",
                        log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs'))
    log.config({"epochs": N_EPOCHS, "n_pts": 8000, "n_boundary": 3000,
                "koch_level": 4, "device": str(DEVICE)})

    for name, mdl, lam_val, hist in [
        ('Uniform', model_uniform, lam_uniform, hist_uniform),
        ('Point-swap', model_swap, lam_swap, hist_swap),
        ('Importance-weighted', model_iw, lam_iw, hist_iw)
    ]:
        res_vals = compute_residual(mdl, xy_eval, d_eval, verts)
        print(f'  {name:20s}: λ = {lam_val:.4f}, mean |∇²u + λu| = {res_vals.mean():.4f}')
        log.metric(f"{name}/lambda", round(lam_val, 4))
        log.metric(f"{name}/mean_residual", round(float(res_vals.mean()), 4))

    # count catastrophic spikes: loss > 100 in second half of training
    for name, hist in [('Uniform', hist_uniform), ('Point-swap', hist_swap),
                       ('Importance-weighted', hist_iw)]:
        phys = np.array(hist['phys'])
        half = len(phys) // 2
        spikes = int(np.sum(phys[half:] > 100.0))
        print(f'  {name:20s}: {spikes} catastrophic spikes (loss>100, 2nd half)')
        log.metric(f"{name}/catastrophic_spikes", spikes)

    # save loss curves to log
    for name, hist in [('Uniform', hist_uniform), ('Point-swap', hist_swap),
                       ('Importance-weighted', hist_iw)]:
        # subsample to keep log file reasonable (~200 points per curve)
        phys = hist['phys']
        step = max(1, len(phys) // 200)
        log.step({f"{name}/phys_loss": [round(p, 6) for p in phys[::step]],
                  f"{name}/lambda": [round(l, 4) for l in hist['lam'][::step]]})

    log.metric("total_time_seconds", round(elapsed, 1))
    log.save()

    # plot
    outdir = os.path.dirname(__file__)
    plot_three_way(
        models=[model_uniform, model_swap, model_iw],
        lams=[lam_uniform, lam_swap, lam_iw],
        histories=[hist_uniform, hist_swap, hist_iw],
        labels=['Uniform', 'Point-swap', 'Importance-weighted'],
        verts=verts,
        outpath=os.path.join(outdir, 'multiscale_koch.png')
    )
