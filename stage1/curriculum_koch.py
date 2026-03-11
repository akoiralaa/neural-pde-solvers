# curriculum training for Koch snowflake eigenfunction
# start on coarse Koch (level 2), transfer to level 3, then level 4
# + gradual point injection + per-point loss normalization

import torch
import torch.nn as nn
import numpy as np
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import time

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ---- geometry ----

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


def nearest_edge_length(xy, verts):
    """For each point, find the length of the nearest boundary edge."""
    v = verts[:-1]; w = verts[1:]
    segs = w - v
    seg_lengths = np.linalg.norm(segs, axis=1)
    ab_sq = (segs**2).sum(axis=1)

    nearest_len = np.zeros(len(xy))
    min_d = np.full(len(xy), 1e10)

    for s in range(0, len(segs), 500):
        e = min(s + 500, len(segs))
        ab_c, v_c, ab_sq_c = segs[s:e], v[s:e], ab_sq[s:e]
        sl_c = seg_lengths[s:e]
        ap = xy[:, None, :] - v_c[None, :, :]
        t = np.clip((ap * ab_c[None]).sum(2) / np.maximum(ab_sq_c[None], 1e-12), 0, 1)
        proj = v_c[None] + t[:, :, None] * ab_c[None]
        dists = np.sqrt(((xy[:, None] - proj)**2).sum(2))

        # for each point, find which edge in this chunk is nearest
        chunk_min_idx = dists.argmin(axis=1)
        chunk_min_d = dists[np.arange(len(xy)), chunk_min_idx]

        update = chunk_min_d < min_d
        min_d[update] = chunk_min_d[update]
        nearest_len[update] = sl_c[chunk_min_idx[update]]

    return nearest_len


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


def compute_residual(model, xy_np, d_np):
    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
    model.eval()
    u = d_t * model(xy_t)
    lap = compute_laplacian(u, xy_t)
    res = (lap + model.lam * u).detach().cpu().numpy()
    model.train()
    return np.abs(res)


# ---- curriculum training ----

def train_curriculum(side=2.0, levels=[2, 3, 4],
                     epochs_per_level=[6000, 5000, 5000],
                     n_interior=6000, n_boundary=3000,
                     inject_rate=0.15, lr=1e-3):
    """
    Curriculum: train on coarse Koch, transfer to finer levels.
    At each level transition, keep 85% of collocation points and
    inject 15% new points near high-residual regions.
    Per-point loss normalization by local edge length.
    """
    rng = np.random.default_rng(42)

    model = FourierNet(n_freqs=64, width=160, depth=5, scale=4.0).to(DEVICE)

    # init lambda near expected value (~20 for Koch with side=2)
    lam_init = 13.0
    model.raw_lam.data.fill_(np.log(np.exp(lam_init - 0.5) - 1))

    history = {'loss': [], 'phys': [], 'lam': [], 'level_starts': []}
    total_ep = 0

    for stage, level in enumerate(levels):
        verts = koch_at_level(level, side)
        n_edges = len(verts) - 1
        print(f'\n{"="*60}')
        print(f'CURRICULUM LEVEL {level} ({n_edges} edges)')
        print(f'{"="*60}')

        history['level_starts'].append(total_ep)

        xy_bdy = torch.tensor(sample_boundary(verts, n_boundary), dtype=torch.float32, device=DEVICE)

        # sample interior points
        if stage == 0:
            # first level: all uniform
            xy_np = sample_interior(verts, n_interior, rng)
        else:
            # transfer: keep (1-inject_rate) old points that are still inside new domain
            path_new = Path(verts)
            still_inside = path_new.contains_points(xy_np)
            xy_keep = xy_np[still_inside]

            n_keep = int(n_interior * (1 - inject_rate))
            if len(xy_keep) > n_keep:
                xy_keep = xy_keep[:n_keep]

            # inject new points near high-residual areas
            n_inject = n_interior - len(xy_keep)
            xy_probe = sample_interior(verts, n_interior * 2, rng)
            d_probe = dist_to_boundary(xy_probe, verts)
            res = compute_residual(model, xy_probe, d_probe)

            weights = res**2
            weights /= weights.sum() + 1e-10
            idx = rng.choice(len(xy_probe), size=n_inject, p=weights, replace=True)
            xy_inject = xy_probe[idx] + rng.normal(0, 0.003, (n_inject, 2))
            inside_mask = path_new.contains_points(xy_inject)
            xy_inject = xy_inject[inside_mask]

            # pad with uniform if needed
            if len(xy_keep) + len(xy_inject) < n_interior:
                n_pad = n_interior - len(xy_keep) - len(xy_inject)
                xy_pad = sample_interior(verts, n_pad, rng)
                xy_np = np.vstack([xy_keep, xy_inject, xy_pad])
            else:
                xy_np = np.vstack([xy_keep, xy_inject[:n_interior - len(xy_keep)]])

            print(f'  Transferred {len(xy_keep)} pts, injected {len(xy_inject)} near residual')

        d_np = dist_to_boundary(xy_np, verts)
        edge_len = nearest_edge_length(xy_np, verts)

        # per-point normalization weights: larger weight for points near large features
        # this prevents tiny-feature points from dominating the loss
        norm_weights = edge_len / (edge_len.mean() + 1e-10)
        norm_weights = np.clip(norm_weights, 0.1, 5.0)  # clip extremes
        norm_w_t = torch.tensor(norm_weights, dtype=torch.float32, device=DEVICE)

        d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
        xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)

        # lr decays across curriculum levels
        level_lr = lr * (0.7 ** stage)
        opt = torch.optim.Adam(
            [{'params': [p for n, p in model.named_parameters() if n != 'raw_lam'], 'lr': level_lr},
             {'params': [model.raw_lam], 'lr': level_lr * 5}]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_per_level[stage], eta_min=1e-6
        )

        best_loss, best_state, best_lam = float('inf'), None, model.lam.item()

        for ep in range(epochs_per_level[stage]):
            # gradual injection: every 2000 epochs, replace 10% of points
            if ep > 0 and ep % 2000 == 0:
                n_replace = int(n_interior * 0.1)
                n_keep_inner = n_interior - n_replace

                # compute residual on current points
                res_curr = compute_residual(model, xy_np, d_np)

                # keep top-(1-10%) lowest residual points (well-fitted)
                keep_idx = np.argsort(res_curr)[:n_keep_inner]
                xy_keep_inner = xy_np[keep_idx]

                # inject new points from high-residual probe
                xy_probe_inner = sample_interior(verts, n_interior, rng)
                d_probe_inner = dist_to_boundary(xy_probe_inner, verts)
                res_probe = compute_residual(model, xy_probe_inner, d_probe_inner)

                weights_inner = res_probe**2
                weights_inner /= weights_inner.sum() + 1e-10
                idx_inner = rng.choice(len(xy_probe_inner), size=n_replace,
                                       p=weights_inner, replace=True)
                xy_new = xy_probe_inner[idx_inner] + rng.normal(0, 0.002, (n_replace, 2))
                path_curr = Path(verts)
                inside_new = path_curr.contains_points(xy_new)
                xy_new = xy_new[inside_new]

                if len(xy_new) > 0:
                    xy_np = np.vstack([xy_keep_inner, xy_new])[:n_interior]
                    d_np = dist_to_boundary(xy_np, verts)
                    edge_len = nearest_edge_length(xy_np, verts)
                    norm_weights = np.clip(edge_len / (edge_len.mean() + 1e-10), 0.1, 5.0)
                    norm_w_t = torch.tensor(norm_weights, dtype=torch.float32, device=DEVICE)
                    d_t = torch.tensor(d_np, dtype=torch.float32, device=DEVICE)
                    xy_t = torch.tensor(xy_np, dtype=torch.float32, device=DEVICE, requires_grad=True)

            opt.zero_grad()
            lam = model.lam

            raw = model(xy_t)
            u = d_t * raw

            lap = compute_laplacian(u, xy_t)
            residual = (lap + lam * u) / (lam.detach() + 1.0)

            # per-point weighted loss
            physics_loss = (norm_w_t * residual**2).mean()

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

            if ep % 1000 == 0 or ep == epochs_per_level[stage] - 1:
                print(f'  ep {total_ep:5d} | loss {loss.item():.5f} | '
                      f'phys {physics_loss.item():.6f} | u² {u_sq.item():.4f} | λ {lam.item():.4f}')

            total_ep += 1

        if best_state:
            model.load_state_dict(best_state)
        print(f'  Level {level} best λ = {best_lam:.4f}')

    print(f'\nFinal λ = {best_lam:.4f}')
    return model, best_lam, history


def plot_results(model_curriculum, lam_curriculum, history,
                 model_baseline, lam_baseline, hist_baseline):
    """Compare curriculum vs baseline (single-level-4 training)."""
    verts = koch_at_level(4, side=2.0)
    path = Path(verts)
    xmin, ymin = verts.min(0) - 0.03
    xmax, ymax = verts.max(0) + 0.03
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

    tri = Delaunay(xy_in)

    results = {}
    for name, mdl, lam_val in [('baseline', model_baseline, lam_baseline),
                                 ('curriculum', model_curriculum, lam_curriculum)]:
        mdl.eval()
        with torch.no_grad():
            u = (d_t * mdl(xy_t)).cpu().numpy()
        if np.sum(u) < 0:
            u = -u
        u_norm = u / (np.abs(u).max() + 1e-10)

        res_vals = compute_residual(mdl, xy_in, d_in)
        results[name] = {'u': u_norm, 'residual': res_vals, 'lam': lam_val}

    fig = plt.figure(figsize=(22, 14))

    # row 1: eigenfunctions
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_trisurf(xy_in[:, 0], xy_in[:, 1], results['baseline']['u'], triangles=tri.simplices, cmap='viridis', edgecolor='none', alpha=0.95)
    ax1.set_title(f'Baseline (single level 4)\nλ ≈ {results["baseline"]["lam"]:.3f}', fontsize=12)
    ax1.view_init(elev=35, azim=-55)
    ax1.plot(verts[:, 0], verts[:, 1], results['baseline']['u'].min() - 0.05, 'k-', lw=0.3, alpha=0.3)

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_trisurf(xy_in[:, 0], xy_in[:, 1], results['curriculum']['u'], triangles=tri.simplices, cmap='viridis', edgecolor='none', alpha=0.95)
    ax2.set_title(f'Curriculum (2→3→4)\nλ ≈ {results["curriculum"]["lam"]:.3f}', fontsize=12)
    ax2.view_init(elev=35, azim=-55)
    ax2.plot(verts[:, 0], verts[:, 1], results['curriculum']['u'].min() - 0.05, 'k-', lw=0.3, alpha=0.3)

    # row 1 col 3: training curves
    ax3 = fig.add_subplot(233)
    w = 200
    # baseline
    if len(hist_baseline['phys']) > w:
        smooth_b = np.convolve(hist_baseline['phys'], np.ones(w)/w, mode='valid')
        ax3.semilogy(smooth_b, color='blue', linewidth=2, label='Baseline', alpha=0.8)
    # curriculum
    if len(history['phys']) > w:
        smooth_c = np.convolve(history['phys'], np.ones(w)/w, mode='valid')
        ax3.semilogy(smooth_c, color='red', linewidth=2, label='Curriculum', alpha=0.8)
    for ls in history['level_starts'][1:]:
        ax3.axvline(ls, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Physics Loss (smoothed)')
    ax3.set_title('Training Loss Comparison')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # row 2: residual heatmaps
    vmax = np.percentile(results['baseline']['residual'], 95)

    ax4 = fig.add_subplot(234)
    sc1 = ax4.scatter(xy_in[:, 0], xy_in[:, 1], c=results['baseline']['residual'],
                       s=1, cmap='hot', vmin=0, vmax=vmax)
    ax4.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.3)
    ax4.set_title('Residual: baseline'); ax4.set_aspect('equal')
    plt.colorbar(sc1, ax=ax4, shrink=0.7)

    ax5 = fig.add_subplot(235)
    sc2 = ax5.scatter(xy_in[:, 0], xy_in[:, 1], c=results['curriculum']['residual'],
                       s=1, cmap='hot', vmin=0, vmax=vmax)
    ax5.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=0.3)
    ax5.set_title('Residual: curriculum'); ax5.set_aspect('equal')
    plt.colorbar(sc2, ax=ax5, shrink=0.7)

    # row 2 col 3: lambda convergence
    ax6 = fig.add_subplot(236)
    ax6.plot(hist_baseline['lam'], color='blue', alpha=0.5, label='Baseline λ')
    ax6.plot(history['lam'], color='red', alpha=0.5, label='Curriculum λ')
    for ls in history['level_starts'][1:]:
        ax6.axvline(ls, color='green', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Epoch'); ax6.set_ylabel('λ')
    ax6.set_title('Eigenvalue Convergence')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    # stats
    mean_b = results['baseline']['residual'].mean()
    mean_c = results['curriculum']['residual'].mean()
    improvement = mean_b / (mean_c + 1e-10)

    fig.suptitle('Curriculum Training vs Baseline — Koch Snowflake Eigenfunction', fontsize=15, y=0.98)

    fig.text(0.5, 0.02,
             f'Mean residual — Baseline: {mean_b:.4f}  |  Curriculum: {mean_c:.4f}  |  '
             f'Improvement: {improvement:.1f}x',
             ha='center', fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig, improvement


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    t_start = time.time()

    # baseline: train directly on level 4
    print('\n' + '='*60)
    print('BASELINE: direct level-4 training')
    print('='*60)

    # use the same FourierNet but train only on level 4
    from multiscale_koch import train_adaptive
    verts4 = koch_at_level(4, side=2.0)
    model_baseline, lam_baseline, hist_baseline = train_adaptive(
        verts4, lam_init=13.0,
        n_phases=1, epochs_per_phase=8000,
        n_uniform=6000, n_adaptive=0, n_boundary=3000, lr=1e-3
    )

    # curriculum: 2 → 3 → 4
    print('\n' + '='*60)
    print('CURRICULUM: level 2 → 3 → 4')
    print('='*60)
    model_curriculum, lam_curriculum, hist_curriculum = train_curriculum(
        side=2.0, levels=[2, 3, 4],
        epochs_per_level=[6000, 5000, 5000],
        n_interior=6000, n_boundary=3000,
        inject_rate=0.15, lr=1e-3
    )

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed:.1f}s')

    # compare residuals
    print('\n--- Final residual comparison ---')
    xy_eval = sample_interior(verts4, 10000, np.random.default_rng(99))
    d_eval = dist_to_boundary(xy_eval, verts4)
    res_b = compute_residual(model_baseline, xy_eval, d_eval)
    res_c = compute_residual(model_curriculum, xy_eval, d_eval)
    print(f'  Baseline:    mean |∇²u + λu| = {res_b.mean():.4f}')
    print(f'  Curriculum:  mean |∇²u + λu| = {res_c.mean():.4f}')
    print(f'  Improvement: {res_b.mean() / (res_c.mean() + 1e-10):.1f}x')

    outdir = '/Users/abhiekoirala/Desktop/neural-pde-solvers/stage1'
    fig, improvement = plot_results(
        model_curriculum, lam_curriculum, hist_curriculum,
        model_baseline, lam_baseline, hist_baseline
    )
    fig.savefig(f'{outdir}/curriculum_koch.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {outdir}/curriculum_koch.png')
