# adaptive multi-scale PINN for Koch snowflake eigenfunction
# strategy: train base solution, compute residual map, adaptively
# resample collocation points where residual is highest, retrain
# repeat at each fractal refinement level

import torch
import torch.nn as nn
import numpy as np
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time

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
            # resample every 2000 epochs within phase
            if ep > 0 and ep % 2000 == 0:
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


def plot_comparison(model_single, model_adaptive, lam_s, lam_a, verts, history_a):
    """Compare single-scale vs adaptive multi-scale."""
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

    # compute solutions and residuals
    results = {}
    for name, mdl in [('single', model_single), ('adaptive', model_adaptive)]:
        mdl.eval()
        with torch.no_grad():
            u = (d_t * mdl(xy_t)).cpu().numpy()
        if np.sum(u) < 0:
            u = -u
        u_norm = u / (np.abs(u).max() + 1e-10)

        res_vals = compute_residual(mdl, xy_in, d_in, verts)

        results[name] = {'u': u_norm, 'residual': res_vals}

    fig = plt.figure(figsize=(20, 14))

    # row 1: eigenfunctions
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_trisurf(tri, results['single']['u'], cmap='viridis', edgecolor='none', alpha=0.95)
    ax1.set_title(f'Single-scale (λ ≈ {lam_s:.3f})', fontsize=12)
    ax1.view_init(elev=35, azim=-55)
    z_floor = float(results['single']['u'].min()) - 0.05
    ax1.plot(verts[:, 0], verts[:, 1], z_floor, 'k-', linewidth=0.3, alpha=0.3)

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_trisurf(tri, results['adaptive']['u'], cmap='viridis', edgecolor='none', alpha=0.95)
    ax2.set_title(f'Adaptive multi-scale (λ ≈ {lam_a:.3f})', fontsize=12)
    ax2.view_init(elev=35, azim=-55)
    z_floor = float(results['adaptive']['u'].min()) - 0.05
    ax2.plot(verts[:, 0], verts[:, 1], z_floor, 'k-', linewidth=0.3, alpha=0.3)

    # row 1, col 3: residual comparison (log scale)
    ax3 = fig.add_subplot(233, projection='3d')
    log_res_s = np.log10(results['single']['residual'] + 1e-10)
    log_res_a = np.log10(results['adaptive']['residual'] + 1e-10)
    ax3.plot_trisurf(tri, log_res_s, cmap='hot', edgecolor='none', alpha=0.7, label='Single')
    ax3.set_title('Residual: single (hot)', fontsize=12)
    ax3.view_init(elev=35, azim=-55)

    # row 2: residual maps (2D heatmaps)
    ax4 = fig.add_subplot(234)
    sc1 = ax4.scatter(xy_in[:, 0], xy_in[:, 1], c=results['single']['residual'],
                       s=1, cmap='hot', vmin=0, vmax=np.percentile(results['single']['residual'], 95))
    ax4.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.5)
    ax4.set_title('Residual map: single-scale'); ax4.set_aspect('equal')
    plt.colorbar(sc1, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(235)
    sc2 = ax5.scatter(xy_in[:, 0], xy_in[:, 1], c=results['adaptive']['residual'],
                       s=1, cmap='hot', vmin=0, vmax=np.percentile(results['single']['residual'], 95))
    ax5.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.5)
    ax5.set_title('Residual map: adaptive'); ax5.set_aspect('equal')
    plt.colorbar(sc2, ax=ax5, shrink=0.7)

    # row 2, col 3: training curves with phase boundaries
    ax6 = fig.add_subplot(236)
    ax6.semilogy(history_a['phys'], alpha=0.3, color='blue')
    # smooth
    w = 200
    if len(history_a['phys']) > w:
        smooth = np.convolve(history_a['phys'], np.ones(w)/w, mode='valid')
        ax6.semilogy(smooth, color='blue', linewidth=2, label='Physics loss')
    for ps in history_a['phase_starts'][1:]:
        ax6.axvline(ps, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Epoch'); ax6.set_ylabel('Physics Loss')
    ax6.set_title('Adaptive training (red = resample)'); ax6.grid(True, alpha=0.3)
    ax6.legend()

    # stats text
    mean_res_s = results['single']['residual'].mean()
    mean_res_a = results['adaptive']['residual'].mean()
    improvement = mean_res_s / (mean_res_a + 1e-10)

    fig.text(0.5, 0.02,
             f'Mean residual — Single: {mean_res_s:.4f}  |  Adaptive: {mean_res_a:.4f}  |  '
             f'Improvement: {improvement:.1f}x',
             ha='center', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Multi-Scale Adaptive PINN — Koch Snowflake Eigenfunction', fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    t_start = time.time()

    # Koch snowflake at level 4 (finest)
    verts = koch_at_level(4, side=2.0)
    print(f'Koch boundary: {len(verts)-1} edges')

    # 1) single-scale baseline (same as eigenfunction_pinn.py)
    print('\n' + '='*60)
    print('SINGLE-SCALE BASELINE')
    print('='*60)
    model_single, lam_single, hist_single = train_adaptive(
        verts, lam_init=13.0,
        n_phases=1, epochs_per_phase=8000,  # single phase = no adaptive resampling
        n_uniform=6000, n_adaptive=0, n_boundary=3000, lr=1e-3
    )

    # 2) adaptive multi-scale
    print('\n' + '='*60)
    print('ADAPTIVE MULTI-SCALE')
    print('='*60)
    model_adaptive, lam_adaptive, hist_adaptive = train_adaptive(
        verts, lam_init=13.0,
        n_phases=4, epochs_per_phase=5000,  # 4 phases, 20k total epochs
        n_uniform=4000, n_adaptive=4000, n_boundary=3000, lr=1e-3
    )

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed:.1f}s')

    # residual comparison
    print('\n--- Residual comparison ---')
    xy_eval = sample_interior(verts, 10000, np.random.default_rng(99))
    d_eval = dist_to_boundary(xy_eval, verts)
    res_s = compute_residual(model_single, xy_eval, d_eval, verts)
    res_a = compute_residual(model_adaptive, xy_eval, d_eval, verts)
    print(f'  Single-scale:  mean |∇²u + λu| = {res_s.mean():.4f}')
    print(f'  Adaptive:      mean |∇²u + λu| = {res_a.mean():.4f}')
    print(f'  Improvement:   {res_s.mean() / (res_a.mean() + 1e-10):.1f}x')

    # plot
    outdir = '/Users/abhiekoirala/Desktop/neural-pde-solvers/stage1'
    fig = plot_comparison(model_single, model_adaptive, lam_single, lam_adaptive,
                          verts, hist_adaptive)
    fig.savefig(f'{outdir}/multiscale_koch.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {outdir}/multiscale_koch.png')
