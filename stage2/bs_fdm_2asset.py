# 2-asset Black-Scholes FDM solver — baseline for PINN comparison
# solves the 2D BS PDE for a rainbow option: payoff = (max(S1, S2) - K)+
# fully implicit Euler in time, central differences in space

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time


def bs_2asset_fdm(K=100, T=1.0, r=0.05, sigma1=0.2, sigma2=0.3, rho=0.5,
                  S1_max=300, S2_max=300, N1=60, N2=60, Nt=300):
    ds1 = S1_max / N1
    ds2 = S2_max / N2
    dt = T / Nt
    s1 = np.linspace(0, S1_max, N1 + 1)
    s2 = np.linspace(0, S2_max, N2 + 1)

    # terminal payoff
    S1g, S2g = np.meshgrid(s1, s2, indexing='ij')
    V = np.maximum(np.maximum(S1g, S2g) - K, 0.0)

    n_int = (N1 - 1) * (N2 - 1)
    idx = lambda i, j: (i - 1) * (N2 - 1) + (j - 1)  # 2D -> 1D index

    # precompute the implicit matrix (constant coefficients at each grid point)
    rows, cols, vals = [], [], []

    def add(r_, c_, v_):
        rows.append(r_); cols.append(c_); vals.append(v_)

    for i in range(1, N1):
        for j in range(1, N2):
            k = idx(i, j)
            si, sj = s1[i], s2[j]

            a1 = 0.5 * sigma1**2 * si**2 / ds1**2
            a2 = 0.5 * sigma2**2 * sj**2 / ds2**2
            a12 = 0.25 * rho * sigma1 * sigma2 * si * sj / (ds1 * ds2)
            b1 = 0.5 * r * si / ds1
            b2 = 0.5 * r * sj / ds2

            # center: 1 + dt*(a1 + a2 + r/2 + r/2) = 1 + dt*(a1+a2+r)
            add(k, k, 1.0 + dt * (2*a1 + 2*a2 + r))

            # S1 neighbors
            if i < N1 - 1:
                add(k, idx(i+1, j), -dt * (a1 + b1))
            if i > 1:
                add(k, idx(i-1, j), -dt * (a1 - b1))

            # S2 neighbors
            if j < N2 - 1:
                add(k, idx(i, j+1), -dt * (a2 + b2))
            if j > 1:
                add(k, idx(i, j-1), -dt * (a2 - b2))

            # cross derivative neighbors
            if i < N1 - 1 and j < N2 - 1:
                add(k, idx(i+1, j+1), -dt * a12)
            if i > 1 and j > 1:
                add(k, idx(i-1, j-1), -dt * a12)
            if i < N1 - 1 and j > 1:
                add(k, idx(i+1, j-1), dt * a12)
            if i > 1 and j < N2 - 1:
                add(k, idx(i-1, j+1), dt * a12)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))

    # time stepping backward
    for step in range(Nt):
        t_now = T - (step + 1) * dt

        # RHS = current interior values + boundary contributions
        rhs = V[1:N1, 1:N2].flatten().copy()

        # add boundary contributions where neighbors are on the boundary
        for i in range(1, N1):
            for j in range(1, N2):
                k = idx(i, j)
                si, sj = s1[i], s2[j]
                a1 = 0.5 * sigma1**2 * si**2 / ds1**2
                a2 = 0.5 * sigma2**2 * sj**2 / ds2**2
                a12 = 0.25 * rho * sigma1 * sigma2 * si * sj / (ds1 * ds2)
                b1 = 0.5 * r * si / ds1
                b2 = 0.5 * r * sj / ds2

                if i == N1 - 1:  # neighbor at i+1 = N1 is boundary
                    rhs[k] += dt * (a1 + b1) * V[N1, j]
                if i == 1:  # neighbor at i-1 = 0 is boundary
                    rhs[k] += dt * (a1 - b1) * V[0, j]
                if j == N2 - 1:
                    rhs[k] += dt * (a2 + b2) * V[i, N2]
                if j == 1:
                    rhs[k] += dt * (a2 - b2) * V[i, 0]

                # cross boundary contributions
                if i == N1 - 1 and j == N2 - 1:
                    rhs[k] += dt * a12 * V[N1, N2]
                if i == 1 and j == 1:
                    rhs[k] += dt * a12 * V[0, 0]
                if i == N1 - 1 and j == 1:
                    rhs[k] -= dt * a12 * V[N1, 0]
                if i == 1 and j == N2 - 1:
                    rhs[k] -= dt * a12 * V[0, N2]

        V_int = spsolve(A, rhs)
        V[1:N1, 1:N2] = V_int.reshape(N1 - 1, N2 - 1)

        # update boundary conditions
        V[0, :] = np.maximum(s2 - K, 0) * np.exp(-r * (T - t_now))
        V[:, 0] = np.maximum(s1 - K, 0) * np.exp(-r * (T - t_now))
        V[N1, :] = S1_max - K * np.exp(-r * t_now)
        V[:, N2] = S2_max - K * np.exp(-r * t_now)

    return s1, s2, V


def plot_surface(s1, s2, V, K=100):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    S1g, S2g = np.meshgrid(s1, s2, indexing='ij')
    im = axes[0].pcolormesh(S1g, S2g, V, shading='auto', cmap='viridis')
    axes[0].set_xlabel('S1')
    axes[0].set_ylabel('S2')
    axes[0].set_title(f'Rainbow Option Price (K={K})')
    plt.colorbar(im, ax=axes[0], label='V(S1, S2)')
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(0, 200)

    j_atm = np.argmin(np.abs(s2 - K))
    axes[1].plot(s1, V[:, j_atm], linewidth=2)
    axes[1].set_xlabel('S1')
    axes[1].set_ylabel('V')
    axes[1].set_title(f'Price Slice at S2 = {s2[j_atm]:.0f}')
    axes[1].axvline(K, color='gray', linestyle='--', alpha=0.5, label=f'K={K}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 200)

    plt.tight_layout()
    plt.savefig('bs_fdm_surface.png', dpi=150)
    print('Saved: bs_fdm_surface.png')


if __name__ == '__main__':
    t0 = time.time()
    print('Running 2-asset BS FDM solver...')
    s1, s2, V = bs_2asset_fdm()
    elapsed = time.time() - t0

    i_atm = np.argmin(np.abs(s1 - 100))
    j_atm = np.argmin(np.abs(s2 - 100))
    print(f'ATM price V(100, 100) = {V[i_atm, j_atm]:.4f}')
    print(f'Grid: {len(s1)}x{len(s2)}, time steps: 300')
    print(f'Wall clock: {elapsed:.2f}s')

    plot_surface(s1, s2, V)
