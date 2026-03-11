import torch
import torch.nn as nn
import numpy as np
from scipy.special import jn_zeros
from matplotlib.path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------- geometry ---------------

def koch_snowflake(order=4, side=2.0):
    # scaled so the domain is bigger -> smaller eigenvalues -> easier training
    h = np.sqrt(3) / 2 * side
    cx, cy = 0.0, 0.0
    verts = np.array([
        [cx, cy + 2*h/3],
        [cx - side/2, cy - h/3],
        [cx + side/2, cy - h/3],
        [cx, cy + 2*h/3]
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

def star_polygon(n_points=5, outer_r=1.0, inner_r=0.38):
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 2*n_points + 1)
    radii = np.empty(2*n_points + 1)
    radii[0::2] = outer_r
    radii[1::2] = inner_r
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def circle_verts(n=200):
    theta = np.linspace(0, 2*np.pi, n+1)
    return np.column_stack([np.cos(theta), np.sin(theta)])

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

def dist_to_boundary_np(xy, verts):
    v = verts[:-1]
    w = verts[1:]
    ab = w - v
    ab_sq = (ab**2).sum(axis=1)
    min_d = np.full(len(xy), 1e10)
    chunk = 500
    for start in range(0, len(ab), chunk):
        end = min(start + chunk, len(ab))
        ab_c = ab[start:end]
        v_c = v[start:end]
        ab_sq_c = ab_sq[start:end]
        ap = xy[:, None, :] - v_c[None, :, :]
        t = (ap * ab_c[None, :, :]).sum(axis=2) / np.maximum(ab_sq_c[None, :], 1e-12)
        t = np.clip(t, 0, 1)
        proj = v_c[None, :, :] + t[:, :, None] * ab_c[None, :, :]
        dists = np.sqrt(((xy[:, None, :] - proj)**2).sum(axis=2))
        min_d = np.minimum(min_d, dists.min(axis=1))
    return min_d

# --------------- network with Fourier features ---------------

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, n_freqs=32, scale=4.0):
        super().__init__()
        B = torch.randn(in_dim, n_freqs) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B  # (N, n_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class EigNet(nn.Module):
    def __init__(self, n_freqs=32, width=128, depth=4, scale=4.0):
        super().__init__()
        self.ff = FourierFeatures(2, n_freqs, scale)
        in_dim = n_freqs * 2
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        # learnable eigenvalue -- use raw param, apply softplus in property
        self.raw_lam = nn.Parameter(torch.tensor(1.5))

    @property
    def lam(self):
        return nn.functional.softplus(self.raw_lam) + 0.5

    def forward(self, xy):
        feat = self.ff(xy)
        return self.net(feat).squeeze(-1)

# --------------- training ---------------

def train_pinn(verts, domain_name, lam_init=5.0,
               n_interior=5000, n_boundary=2000,
               epochs=6000, lr=1e-3):
    rng = np.random.default_rng(42)

    xy_bdy_np = sample_boundary(verts, n_boundary)
    xy_bdy_t = torch.tensor(xy_bdy_np, dtype=torch.float32, device=device)

    model = EigNet(n_freqs=48, width=128, depth=5, scale=3.0).to(device)

    # initialize lambda near expected value
    # softplus(raw) + 0.5 = lam_init => raw = softplus_inv(lam_init - 0.5)
    target_sp = lam_init - 0.5
    raw_init = np.log(np.exp(target_sp) - 1) if target_sp > 0 else 1.0
    model.raw_lam.data.fill_(raw_init)
    print(f"[{domain_name}] lambda init: {model.lam.item():.3f}")

    # separate lr for eigenvalue -- higher so it can move
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if n != 'raw_lam'], 'lr': lr},
        {'params': [model.raw_lam], 'lr': lr * 5}
    ]
    opt = torch.optim.Adam(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_state = None

    for ep in range(epochs):
        # resample interior periodically
        if ep % 1000 == 0:
            xy_int_np = sample_interior(verts, n_interior, rng)
            d_int_np = dist_to_boundary_np(xy_int_np, verts)
            d_int_t = torch.tensor(d_int_np, dtype=torch.float32, device=device)
            xy_int_t = torch.tensor(xy_int_np, dtype=torch.float32, device=device, requires_grad=True)

        opt.zero_grad()

        # forward on interior with hard BC
        raw = model(xy_int_t)
        u = d_int_t * raw

        # laplacian
        grad_u = torch.autograd.grad(u.sum(), xy_int_t, create_graph=True)[0]
        ux, uy = grad_u[:, 0], grad_u[:, 1]
        uxx = torch.autograd.grad(ux.sum(), xy_int_t, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(uy.sum(), xy_int_t, create_graph=True)[0][:, 1]
        lap = uxx + uyy

        lam = model.lam

        # PDE residual: lap(u) + lam * u = 0
        # normalize residual by lam to make it scale-invariant
        residual = (lap + lam * u) / (lam.detach() + 1.0)
        physics_loss = (residual ** 2).mean()

        # normalization: integral(u^2) should be nonzero
        # use log barrier to prevent collapse
        u_sq_mean = (u ** 2).mean()
        # push u_sq_mean toward 0.5 (arbitrary nonzero target)
        norm_loss = (torch.log(u_sq_mean + 1e-8) - np.log(0.5)) ** 2

        # boundary loss (soft, supplement hard BC)
        u_bdy = model(xy_bdy_t)
        bdy_loss = (u_bdy ** 2).mean()

        # weight schedule
        w_phys = 1.0
        w_norm = 5.0
        w_bdy = 0.5  # light since hard BC handles most of it

        loss = w_phys * physics_loss + w_norm * norm_loss + w_bdy * bdy_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        if ep % 1000 == 0:
            # re-create xy_int_t with requires_grad for new points
            xy_int_t = torch.tensor(xy_int_np, dtype=torch.float32, device=device, requires_grad=True)

        with torch.no_grad():
            if loss.item() < best_loss and u_sq_mean.item() > 0.05:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 1000 == 0 or ep == epochs - 1:
            print(f"[{domain_name}] ep {ep:5d} | loss {loss.item():.5f} | "
                  f"phys {physics_loss.item():.6f} | u2 {u_sq_mean.item():.4f} | "
                  f"bdy {bdy_loss.item():.5f} | lam {lam.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"[{domain_name}] final lambda: {model.lam.item():.4f}")
    return model

# --------------- plotting ---------------

def plot_eigenfunction(model, verts, domain_name, resolution=250):
    path = Path(verts)
    xmin, ymin = verts.min(axis=0) - 0.05
    xmax, ymax = verts.max(axis=0) + 0.05
    xi = np.linspace(xmin, xmax, resolution)
    yi = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(xi, yi)
    xy_flat = np.column_stack([X.ravel(), Y.ravel()])
    inside = path.contains_points(xy_flat)
    xy_in = xy_flat[inside]
    d_in = dist_to_boundary_np(xy_in, verts)

    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(xy_in, dtype=torch.float32, device=device)).cpu().numpy()
    u_vals = d_in * raw
    # flip sign so peak is positive (eigenfunctions have arbitrary sign)
    if np.sum(u_vals) < 0:
        u_vals = -u_vals
    umax = np.abs(u_vals).max()
    u_plot = u_vals / umax if umax > 1e-10 else u_vals

    lam_val = model.lam.item()

    tri = mtri.Triangulation(xy_in[:, 0], xy_in[:, 1])
    max_edge = (xmax - xmin) / resolution * 3.5
    ts = tri.triangles
    xt, yt = xy_in[ts, 0], xy_in[ts, 1]
    tri.set_mask((xt.max(1)-xt.min(1) > max_edge) | (yt.max(1)-yt.min(1) > max_edge))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tri, u_plot, cmap='viridis', edgecolor='none',
                           antialiased=True, linewidth=0, alpha=0.95)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.set_title(f'{domain_name} eigenfunction  |  λ ≈ {lam_val:.3f}', fontsize=14)
    ax.view_init(elev=35, azim=-55)
    z_floor = float(u_plot.min()) - 0.05
    ax.plot(verts[:, 0], verts[:, 1], z_floor, 'k-', linewidth=0.6, alpha=0.4)
    fig.colorbar(surf, shrink=0.5, aspect=15, label='u (normalized)')
    plt.tight_layout()
    return fig, lam_val

# --------------- main ---------------

if __name__ == '__main__':
    outdir = '/Users/abhiekoirala/Desktop/neural-pde-solvers/stage1'

    # 1) unit circle validation
    print("\n=== Unit circle validation ===")
    cverts = circle_verts(200)
    exact_lam = jn_zeros(0, 1)[0] ** 2  # ~5.783
    model_c = train_pinn(cverts, "Circle", lam_init=exact_lam,
                         n_interior=5000, n_boundary=1500, epochs=6000, lr=1e-3)
    lam_c = model_c.lam.item()
    print(f"Circle: computed λ = {lam_c:.4f}, exact = {exact_lam:.4f}, "
          f"rel error = {abs(lam_c - exact_lam)/exact_lam*100:.2f}%")
    fig_c, _ = plot_eigenfunction(model_c, cverts, "Unit Circle")
    fig_c.savefig(f'{outdir}/eigenfunction_circle.png', dpi=150, bbox_inches='tight')

    # 2) Koch snowflake (first eigenvalue ~52 for equilateral triangle inscribed in
    # unit circle, but Koch is larger so eigenvalue is smaller)
    print("\n=== Koch snowflake ===")
    koch = koch_snowflake(order=3, side=2.0)
    # with side=2.0, eigenvalue scales as ~52/4 ~ 13
    model_k = train_pinn(koch, "Koch", lam_init=13.0,
                         n_interior=6000, n_boundary=3000, epochs=8000, lr=1e-3)
    lam_k = model_k.lam.item()
    fig_k, _ = plot_eigenfunction(model_k, koch, "Koch Snowflake")
    fig_k.savefig(f'{outdir}/eigenfunction_koch.png', dpi=150, bbox_inches='tight')

    # 3) five-pointed star
    print("\n=== Five-pointed star ===")
    star = star_polygon(5, 1.0, 0.38)
    # star eigenvalue depends on geometry, start with reasonable guess
    model_s = train_pinn(star, "Star", lam_init=15.0,
                         n_interior=6000, n_boundary=2000, epochs=8000, lr=1e-3)
    lam_s = model_s.lam.item()
    fig_s, _ = plot_eigenfunction(model_s, star, "Five-Pointed Star")
    fig_s.savefig(f'{outdir}/eigenfunction_star.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    # combined figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), subplot_kw={'projection': '3d'})
    configs = [
        ("Unit Circle", cverts, model_c),
        ("Koch Snowflake", koch, model_k),
        ("Five-Pointed Star", star, model_s)
    ]
    for ax, (name, v, mdl) in zip(axes, configs):
        path = Path(v)
        pad = 0.03
        xmn, ymn = v.min(axis=0) - pad
        xmx, ymx = v.max(axis=0) + pad
        res = 180
        xi = np.linspace(xmn, xmx, res)
        yi = np.linspace(ymn, ymx, res)
        X, Y = np.meshgrid(xi, yi)
        flat = np.column_stack([X.ravel(), Y.ravel()])
        ins = path.contains_points(flat)
        xyin = flat[ins]
        din = dist_to_boundary_np(xyin, v)
        mdl.eval()
        with torch.no_grad():
            raw = mdl(torch.tensor(xyin, dtype=torch.float32, device=device)).cpu().numpy()
        u = din * raw
        if np.sum(u) < 0:
            u = -u
        um = np.abs(u).max()
        u = u / um if um > 1e-10 else u
        tri = mtri.Triangulation(xyin[:, 0], xyin[:, 1])
        me = (xmx - xmn) / res * 3.5
        ts = tri.triangles
        xt, yt = xyin[ts, 0], xyin[ts, 1]
        tri.set_mask((xt.max(1)-xt.min(1) > me) | (yt.max(1)-yt.min(1) > me))
        ax.plot_trisurf(tri, u, cmap='viridis', edgecolor='none', antialiased=True, alpha=0.95)
        ax.set_title(f'{name}\nλ ≈ {mdl.lam.item():.3f}', fontsize=12)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.view_init(elev=35, azim=-55)

    fig.suptitle('Laplace Eigenfunctions (PINN Solver)', fontsize=15, y=0.98)
    plt.tight_layout()
    fig.savefig(f'{outdir}/eigenfunction_results.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    print("\n=== Final Summary ===")
    print(f"Circle:  λ = {lam_c:.4f} (exact {exact_lam:.4f}, err {abs(lam_c-exact_lam)/exact_lam*100:.1f}%)")
    print(f"Koch:    λ = {lam_k:.4f}")
    print(f"Star:    λ = {lam_s:.4f}")
    print(f"Figures saved to {outdir}/")
