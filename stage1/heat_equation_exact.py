# 1D heat equation exact solver — ground truth for PINN comparison
# PDE: du/dt = alpha * d²u/dx², x in [0,1], u(0,t) = u(1,t) = 0

import numpy as np
import matplotlib.pyplot as plt


def exact_solution_single_mode(x, t, alpha=0.01):  # IC: sin(pi*x)
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def fourier_coefficients(f, n_terms, n_quad=1000):  # bn = 2 * integral(f(x)*sin(n*pi*x), 0, 1)
    x_quad = np.linspace(0, 1, n_quad)
    coeffs = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        integrand = f(x_quad) * np.sin(n * np.pi * x_quad)
        coeffs[n - 1] = 2.0 * np.trapezoid(integrand, x_quad)
    return coeffs


def exact_solution_fourier(x, t, alpha=0.01, n_terms=50, ic=None):  # general IC via fourier series
    if ic is None:
        return exact_solution_single_mode(x, t, alpha)

    coeffs = fourier_coefficients(ic, n_terms)
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim == 0:
        x = x.reshape(1)
    if t.ndim == 0:
        t = t.reshape(1)

    u = np.zeros((len(t), len(x)))
    for i, n in enumerate(range(1, n_terms + 1)):
        u += coeffs[i] * np.outer(
            np.exp(-alpha * n**2 * np.pi**2 * t),
            np.sin(n * np.pi * x)
        )
    return u


def plot_ground_truth(alpha=0.01, T=1.0):
    x = np.linspace(0, 1, 200)
    times = [0.0, 0.1, 0.3, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]  # solution profiles at different times
    for t in times:
        u = exact_solution_single_mode(x, t, alpha)
        ax.plot(x, u, label=f"t = {t:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("1D Heat Equation — Exact Solution (IC: sin(πx))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]  # heatmap over (x, t)
    t_grid = np.linspace(0, T, 200)
    X, T_grid = np.meshgrid(x, t_grid)
    U = exact_solution_single_mode(X, T_grid, alpha)
    im = ax.pcolormesh(X, T_grid, U, shading="auto", cmap="inferno")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Temperature Evolution")
    plt.colorbar(im, ax=ax, label="u(x, t)")

    plt.tight_layout()
    plt.savefig("stage1/ground_truth.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    x_test = np.linspace(0, 1, 100)

    u_0 = exact_solution_single_mode(x_test, 0.0)  # should equal sin(pi*x)
    error = np.max(np.abs(u_0 - np.sin(np.pi * x_test)))
    print(f"IC verification: max error = {error:.2e}")
    assert error < 1e-15

    t_test = np.linspace(0, 1, 100)  # BCs should be 0
    u_left = exact_solution_single_mode(0.0, t_test)
    u_right = exact_solution_single_mode(1.0, t_test)
    bc_error = max(np.max(np.abs(u_left)), np.max(np.abs(u_right)))
    print(f"BC verification: max error = {bc_error:.2e}")
    assert bc_error < 1e-15

    ic_sin = lambda x: np.sin(np.pi * x)  # fourier series should match single mode exactly
    u_fourier = exact_solution_fourier(x_test, np.array([0.5]), alpha=0.01, n_terms=50, ic=ic_sin)
    u_exact = exact_solution_single_mode(x_test, 0.5)
    fourier_error = np.max(np.abs(u_fourier[0] - u_exact))
    print(f"Fourier vs exact: max error = {fourier_error:.2e}")
    assert fourier_error < 1e-10

    print("\nAll verifications passed.")
    plot_ground_truth()
