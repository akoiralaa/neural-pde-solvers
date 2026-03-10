# Heston PDE solver — implicit FD in log-spot coordinates
# prices European call under stochastic volatility
#
# transform x = ln(S), then PDE becomes:
#   ∂V/∂t + (r - v/2)*∂V/∂x + 0.5*v*∂²V/∂x²
#         + ρσv*∂²V/∂x∂v + 0.5*σ²*v*∂²V/∂v²
#         + κ(θ-v)*∂V/∂v - r*V = 0
#
# log transform removes the S² blowup in coefficients
# matrix built once, RHS updated each step for BCs

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def heston_fd(S0, K, T, r, kappa, theta, sigma, rho, v0,
              Nx=150, Nv=60, Nt=500, x_spread=4.0, v_max=None):
    """Solve Heston PDE on (x=ln(S), v) grid, fully implicit."""

    if v_max is None:
        v_max = max(5 * theta, 0.5)

    x_min = np.log(S0) - x_spread
    x_max = np.log(S0) + x_spread
    dx = (x_max - x_min) / Nx
    dv = v_max / Nv
    dt = T / Nt

    x = np.linspace(x_min, x_max, Nx + 1)
    v = np.linspace(0, v_max, Nv + 1)
    S = np.exp(x)

    N_total = (Nx + 1) * (Nv + 1)

    # terminal condition
    V = np.maximum(S[np.newaxis, :] - K, 0.0) * np.ones((Nv + 1, 1))

    def idx(j, i):
        return j * (Nx + 1) + i

    # build implicit matrix once — interior coefficients are time-independent
    rows, cols, vals = [], [], []

    # track which entries are boundary (will update RHS each step)
    boundary_mask = np.zeros(N_total, dtype=bool)

    for j in range(Nv + 1):
        for i in range(Nx + 1):
            k = idx(j, i)

            is_boundary = (i == 0 or i == Nx or j == 0 or j == Nv)
            if is_boundary:
                boundary_mask[k] = True
                rows.append(k); cols.append(k); vals.append(1.0)
                if j == Nv and i > 0 and i < Nx:  # Neumann: V[Nv] = V[Nv-1]
                    rows.append(k); cols.append(idx(j-1, i)); vals.append(-1.0)
            else:
                vj = v[j]
                ax = 0.5 * vj / dx**2
                bx = (r - 0.5 * vj) / (2 * dx)
                av = 0.5 * sigma**2 * vj / dv**2
                bv = kappa * (theta - vj) / (2 * dv)
                axy = rho * sigma * vj / (4 * dx * dv)

                # diagonal
                rows.append(k); cols.append(k)
                vals.append(1.0 + dt * (2*ax + 2*av + r))

                # x neighbors
                rows.append(k); cols.append(idx(j, i-1)); vals.append(-dt * (ax - bx))
                rows.append(k); cols.append(idx(j, i+1)); vals.append(-dt * (ax + bx))

                # v neighbors
                rows.append(k); cols.append(idx(j-1, i)); vals.append(-dt * (av - bv))
                rows.append(k); cols.append(idx(j+1, i)); vals.append(-dt * (av + bv))

                # cross derivative corners
                rows.append(k); cols.append(idx(j+1, i+1)); vals.append(-dt * axy)
                rows.append(k); cols.append(idx(j+1, i-1)); vals.append(dt * axy)
                rows.append(k); cols.append(idx(j-1, i+1)); vals.append(dt * axy)
                rows.append(k); cols.append(idx(j-1, i-1)); vals.append(-dt * axy)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N_total, N_total))

    # time-step: only update boundary RHS
    for n in range(Nt):
        tau = T - (n + 1) * dt  # time remaining after this step

        rhs = V.ravel().copy()

        # update boundary values
        for i in range(Nx + 1):
            rhs[idx(0, i)] = max(S[i] - K * np.exp(-r * tau), 0.0)  # v=0
        for j in range(Nv + 1):
            rhs[idx(j, 0)] = 0.0  # S→0
            rhs[idx(j, Nx)] = S[-1] - K * np.exp(-r * tau)  # S→∞
        for i in range(1, Nx):
            rhs[idx(Nv, i)] = 0.0  # Neumann: solved by matrix row

        V_flat = spsolve(A, rhs)
        V = V_flat.reshape((Nv + 1, Nx + 1))

    return S, v, V


def heston_char_fn_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    """Semi-analytical Heston price via characteristic function (Gil-Pelaez)."""
    import cmath

    def phi(u, S0, T, r, kappa, theta, sigma, rho, v0):
        iu = 1j * u
        d = cmath.sqrt((rho * sigma * iu - kappa)**2 + sigma**2 * (iu + u**2))
        g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)

        C = r * iu * T + (kappa * theta / sigma**2) * (
            (kappa - rho * sigma * iu - d) * T
            - 2 * cmath.log((1 - g * cmath.exp(-d * T)) / (1 - g))
        )
        D = ((kappa - rho * sigma * iu - d) / sigma**2) * (
            (1 - cmath.exp(-d * T)) / (1 - g * cmath.exp(-d * T))
        )
        return cmath.exp(C + D * v0 + iu * cmath.log(S0))

    N_int = 2000
    u_max = 250.0
    du = u_max / N_int

    P1_integral = 0.0
    P2_integral = 0.0
    ln_K = np.log(K)

    for k in range(1, N_int + 1):
        u = k * du
        phi1 = phi(u - 1j, S0, T, r, kappa, theta, sigma, rho, v0)
        phi1 /= phi(-1j, S0, T, r, kappa, theta, sigma, rho, v0)
        P1_integral += (cmath.exp(-1j * u * ln_K) * phi1 / (1j * u)).real * du

        phi2 = phi(u, S0, T, r, kappa, theta, sigma, rho, v0)
        P2_integral += (cmath.exp(-1j * u * ln_K) * phi2 / (1j * u)).real * du

    P1 = 0.5 + P1_integral / np.pi
    P2 = 0.5 + P2_integral / np.pi

    return S0 * P1 - K * np.exp(-r * T) * P2


def interpolate_fd(S_grid, v_grid, V, S0, v0):
    """Bilinear interpolation to get price at (S0, v0)."""
    x0 = np.log(S0)
    x_grid = np.log(S_grid)
    ix = np.searchsorted(x_grid, x0)
    jv = np.searchsorted(v_grid, v0)

    i0, i1 = max(ix - 1, 0), min(ix, len(x_grid) - 1)
    j0, j1 = max(jv - 1, 0), min(jv, len(v_grid) - 1)

    if i0 == i1 or j0 == j1:
        return V[j0, i0]

    wx = (x0 - x_grid[i0]) / (x_grid[i1] - x_grid[i0])
    wv = (v0 - v_grid[j0]) / (v_grid[j1] - v_grid[j0])
    return (V[j0, i0] * (1-wx) * (1-wv) +
            V[j0, i1] * wx * (1-wv) +
            V[j1, i0] * (1-wx) * wv +
            V[j1, i1] * wx * wv)


if __name__ == '__main__':
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

    print('Heston PDE Solver — FD vs Semi-Analytical\n')
    print(f'Params: S0={S0}, K={K}, T={T}, r={r}')
    print(f'        κ={kappa}, θ={theta}, σ={sigma}, ρ={rho}, v0={v0}\n')

    # semi-analytical ground truth
    t0 = time.time()
    price_exact = heston_char_fn_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
    t_exact = time.time() - t0
    print(f'Semi-analytical: {price_exact:.6f}  ({t_exact*1000:.1f}ms)')

    # FD solver
    t0 = time.time()
    S_grid, v_grid, V = heston_fd(S0, K, T, r, kappa, theta, sigma, rho, v0,
                                  Nx=150, Nv=60, Nt=500)
    t_fd = time.time() - t0

    price_fd = interpolate_fd(S_grid, v_grid, V, S0, v0)
    error_pct = abs(price_fd - price_exact) / price_exact * 100

    print(f'FD (implicit):   {price_fd:.6f}  ({t_fd:.1f}s)')
    print(f'Error:           {error_pct:.4f}%')
    print(f'Target:          <1%')
    print(f'{"PASS" if error_pct < 1 else "FAIL"}')

    # strike sweep
    print('\n--- Strike sweep ---')
    for K_test in [80, 90, 100, 110, 120]:
        exact = heston_char_fn_price(S0, K_test, T, r, kappa, theta, sigma, rho, v0)
        S_g, v_g, V_k = heston_fd(S0, K_test, T, r, kappa, theta, sigma, rho, v0,
                                   Nx=150, Nv=60, Nt=500)
        fd_p = interpolate_fd(S_g, v_g, V_k, S0, v0)
        err = abs(fd_p - exact) / exact * 100
        print(f'  K={K_test:3d}: exact={exact:.4f}, FD={fd_p:.4f}, err={err:.3f}%')

    # plot
    fig = plt.figure(figsize=(14, 5))

    ax3d = fig.add_subplot(121, projection='3d')
    S_mesh, v_mesh = np.meshgrid(S_grid, v_grid)
    s_mask = (S_grid >= 50) & (S_grid <= 200)
    ax3d.plot_surface(S_mesh[:, s_mask], v_mesh[:, s_mask], V[:, s_mask],
                      cmap='viridis', alpha=0.8, rcount=50, ccount=50)
    ax3d.set_xlabel('S'); ax3d.set_ylabel('v'); ax3d.set_zlabel('V')
    ax3d.set_title('Heston Call Price V(S, v)')

    ax2 = fig.add_subplot(122)
    v_levels = [0, 5, 15, 30]
    for jj in v_levels:
        if jj < len(v_grid):
            ax2.plot(S_grid, V[jj, :], label=f'v = {v_grid[jj]:.3f}')
    ax2.plot(S_grid, np.maximum(S_grid - K, 0), 'k--', alpha=0.3, label='Intrinsic')
    ax2.set_xlabel('S'); ax2.set_ylabel('V')
    ax2.set_title('Price Slices at Different Variance Levels')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xlim([50, 200])

    plt.tight_layout()
    plt.savefig('heston_fd_surface.png', dpi=150)
    print('\nSaved: heston_fd_surface.png')
