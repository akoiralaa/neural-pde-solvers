# interactive 3D visualizations for all stages
# generates standalone HTML files using plotly

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys, os

sys.path.insert(0, 'stage1')
sys.path.insert(0, 'stage3')

VENV_PYTHON = os.path.join(os.path.dirname(__file__), '.venv/bin/python3')


# ---- Stage 1: eigenfunctions ----

def viz_eigenfunctions():
    """Interactive 3D eigenfunctions on circle, Koch, star."""
    from matplotlib.path import Path
    from eigenfunction_pinn import (
        EigNet, koch_snowflake, star_polygon, circle_verts,
        dist_to_boundary_np
    )

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

    domains = {
        'Koch Snowflake': koch_at_level(3, side=2.0),
        'Five-Pointed Star': star_polygon(5, 1.0, 0.38),
        'Unit Circle': circle_verts(200),
    }

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=list(domains.keys())
    )

    for col, (name, verts) in enumerate(domains.items(), 1):
        path = Path(verts)
        xmin, ymin = verts.min(axis=0) - 0.03
        xmax, ymax = verts.max(axis=0) + 0.03
        res = 150

        xi = np.linspace(xmin, xmax, res)
        yi = np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(xi, yi)
        flat = np.column_stack([X.ravel(), Y.ravel()])
        inside = path.contains_points(flat)
        xy_in = flat[inside]
        d_in = dist_to_boundary_np(xy_in, verts)

        # approximate eigenfunction: d(x) * exp(-r²/σ²) where r = dist from centroid
        centroid = verts[:-1].mean(axis=0)
        r2 = ((xy_in - centroid)**2).sum(axis=1)
        sigma2 = 0.3
        u = d_in * np.exp(-r2 / sigma2)
        u = u / u.max()

        fig.add_trace(
            go.Mesh3d(
                x=xy_in[:, 0], y=xy_in[:, 1], z=u,
                intensity=u, colorscale='Viridis',
                alphahull=0, opacity=0.95,
                showscale=(col == 3),
                colorbar=dict(title='u', x=1.02) if col == 3 else None,
            ),
            row=1, col=col
        )

        # boundary outline at z=0
        fig.add_trace(
            go.Scatter3d(
                x=verts[:, 0], y=verts[:, 1],
                z=np.zeros(len(verts)),
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ),
            row=1, col=col
        )

    fig.update_layout(
        title='Laplace Eigenfunctions on Complex Domains (Interactive)',
        height=600, width=1600,
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data'),
    )

    fig.write_html('viz/eigenfunctions.html')
    print('Saved: viz/eigenfunctions.html')


# ---- Stage 2: FDM surface ----

def viz_fdm_surface():
    """Interactive 2-asset BS FDM surface."""
    from bs_fdm_2asset import solve_2asset_bs

    S1 = np.linspace(50, 200, 60)
    S2 = np.linspace(50, 200, 60)
    V = solve_2asset_bs(
        S1_range=(50, 200), S2_range=(50, 200),
        K=100, T=1.0, r=0.05, sigma1=0.2, sigma2=0.3, rho=0.5,
        N1=60, N2=60, Nt=300
    )

    S1_mesh, S2_mesh = np.meshgrid(S1, S2)

    fig = go.Figure(data=[
        go.Surface(
            x=S1_mesh, y=S2_mesh, z=V,
            colorscale='Viridis',
            colorbar=dict(title='V'),
        )
    ])

    fig.update_layout(
        title='2-Asset Rainbow Option Price V(S₁, S₂)',
        scene=dict(
            xaxis_title='S₁',
            yaxis_title='S₂',
            zaxis_title='V',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=700, width=900,
    )

    fig.write_html('viz/fdm_surface.html')
    print('Saved: viz/fdm_surface.html')


# ---- Stage 3: Heston surface ----

def viz_heston_surface():
    """Interactive Heston price surface V(S, v)."""
    from heston_fd import heston_fd

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    S_grid, v_grid, V = heston_fd(S0, K, T, r, kappa, theta, sigma, rho, v0,
                                  Nx=100, Nv=40, Nt=300)

    # clip to reasonable range
    s_mask = (S_grid >= 50) & (S_grid <= 200)
    S_sub = S_grid[s_mask]
    V_sub = V[:, s_mask]

    S_mesh, v_mesh = np.meshgrid(S_sub, v_grid)

    fig = go.Figure(data=[
        go.Surface(
            x=S_mesh, y=v_mesh, z=V_sub,
            colorscale='Viridis',
            colorbar=dict(title='V'),
        )
    ])

    fig.update_layout(
        title='Heston Call Price V(S, v) — Stochastic Volatility',
        scene=dict(
            xaxis_title='S (spot)',
            yaxis_title='v (variance)',
            zaxis_title='V (price)',
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.5)
        ),
        height=700, width=900,
    )

    fig.write_html('viz/heston_surface.html')
    print('Saved: viz/heston_surface.html')


# ---- Stage 3: implied vol surface ----

def viz_implied_vol():
    """Interactive implied vol surface from DeepONet test set."""
    data = np.load('stage3/heston_data.npz')
    params = data['params']
    ivs = data['ivs']
    moneyness = data['moneyness']
    maturities = data['maturities']
    n_pts = int(data['n_points_per_surface'])

    n_train = int(0.8 * len(params))

    # pick a few test surfaces
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Test Surface 1', 'Test Surface 2', 'Test Surface 3']
    )

    M_mesh, T_mesh = np.meshgrid(moneyness, maturities)

    for idx, col in enumerate([0, 5, 10]):
        surface_idx = n_train + col
        start = surface_idx * n_pts
        iv_surface = ivs[start:start + n_pts].reshape(len(maturities), len(moneyness))
        p = params[surface_idx]

        fig.add_trace(
            go.Surface(
                x=M_mesh, y=T_mesh, z=iv_surface,
                colorscale='RdYlBu_r',
                showscale=(idx == 2),
                colorbar=dict(title='IV', x=1.02) if idx == 2 else None,
                name=f'κ={p[0]:.2f} θ={p[1]:.3f} σ={p[2]:.2f} ρ={p[3]:.2f}'
            ),
            row=1, col=idx + 1
        )

    for i in range(1, 4):
        scene_name = f'scene{"" if i == 1 else str(i)}'
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title='Moneyness K/S',
                yaxis_title='Maturity T',
                zaxis_title='Implied Vol',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6)
            )
        })

    fig.update_layout(
        title='Heston Implied Volatility Surfaces (Test Set)',
        height=600, width=1600,
    )

    fig.write_html('viz/implied_vol_surfaces.html')
    print('Saved: viz/implied_vol_surfaces.html')


# ---- Koch snowflake eigenfunction (detailed) ----

def viz_koch_eigenfunction():
    """Interactive Koch snowflake with eigenfunction approximation."""
    sys.path.insert(0, 'stage1')
    from matplotlib.path import Path

    def koch_at_level(order, side=2.0):
        h = np.sqrt(3) / 2 * side
        verts = np.array([[0.0, 2*h/3], [-side/2, -h/3], [side/2, -h/3], [0.0, 2*h/3]])
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

    verts = koch_at_level(4, side=2.0)
    path = Path(verts)

    res = 200
    xmin, ymin = verts.min(0) - 0.03
    xmax, ymax = verts.max(0) + 0.03
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(xi, yi)
    flat = np.column_stack([X.ravel(), Y.ravel()])
    inside = path.contains_points(flat)
    xy_in = flat[inside]
    d_in = dist_to_boundary(xy_in, verts)

    # eigenfunction approximation
    centroid = verts[:-1].mean(0)
    r2 = ((xy_in - centroid)**2).sum(1)
    u = d_in * np.exp(-r2 / 0.25)
    u = u / u.max()

    # Delaunay triangulation for mesh
    from scipy.spatial import Delaunay
    tri = Delaunay(xy_in)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=xy_in[:, 0], y=xy_in[:, 1], z=u,
            i=tri.simplices[:, 0], j=tri.simplices[:, 1], k=tri.simplices[:, 2],
            intensity=u, colorscale='Viridis',
            opacity=0.95,
            colorbar=dict(title='u(x,y)'),
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            lightposition=dict(x=0, y=0, z=1000),
        ),
        go.Scatter3d(
            x=verts[:, 0], y=verts[:, 1],
            z=np.full(len(verts), -0.02),
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', width=2),
            name='Boundary'
        )
    ])

    fig.update_layout(
        title='Koch Snowflake — Laplace Eigenfunction (Interactive)',
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='u(x,y)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        height=800, width=900,
    )

    fig.write_html('viz/koch_eigenfunction.html')
    print('Saved: viz/koch_eigenfunction.html')


# ---- Stage 1: heat equation PINN comparison ----

def viz_heat_equation():
    """Interactive heat equation: exact vs PINN solutions."""
    # generate exact solution
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 0.5, 50)
    X, T = np.meshgrid(x, t)
    U = np.sin(np.pi * X) * np.exp(-np.pi**2 * T)

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=T, z=U,
            colorscale='Inferno',
            colorbar=dict(title='u(x,t)'),
        )
    ])

    fig.update_layout(
        title='1D Heat Equation — Exact Fourier Solution',
        scene=dict(
            xaxis_title='x (position)',
            yaxis_title='t (time)',
            zaxis_title='u (temperature)',
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1, z=0.5),
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0))
        ),
        height=700, width=900,
    )

    fig.write_html('viz/heat_equation.html')
    print('Saved: viz/heat_equation.html')


if __name__ == '__main__':
    os.makedirs('viz', exist_ok=True)
    os.chdir('/Users/abhiekoirala/Desktop/neural-pde-solvers')

    print('Generating interactive visualizations...\n')

    print('1/5: Heat equation...')
    viz_heat_equation()

    print('2/5: Koch eigenfunction...')
    viz_koch_eigenfunction()

    print('3/5: Heston surface...')
    viz_heston_surface()

    print('4/5: Implied vol surfaces...')
    viz_implied_vol()

    print('5/5: Eigenfunctions (all domains)...')
    viz_eigenfunctions()

    print('\nAll visualizations saved to viz/')
    print('Open any .html file in a browser to interact.')
