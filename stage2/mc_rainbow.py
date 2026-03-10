# Monte Carlo pricer for multi-asset rainbow option
# payoff: (max(S1, ..., Sn) - K)+
# uses correlated GBM with Cholesky decomposition
# works for arbitrary number of assets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


def mc_rainbow(S0, K, T, r, sigmas, corr_matrix, n_paths=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    n_assets = len(S0)
    L = np.linalg.cholesky(corr_matrix)  # correlate the brownian motions

    Z = rng.standard_normal((n_paths, n_assets))
    W = Z @ L.T  # correlated normals

    # GBM terminal values: S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*W)
    sigmas = np.array(sigmas)
    S0 = np.array(S0)
    drift = (r - 0.5 * sigmas**2) * T
    diffusion = sigmas * np.sqrt(T) * W
    S_T = S0 * np.exp(drift + diffusion)  # (n_paths, n_assets)

    payoff = np.maximum(np.max(S_T, axis=1) - K, 0)  # rainbow payoff
    price = np.exp(-r * T) * np.mean(payoff)
    stderr = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)

    return price, stderr, S_T, payoff


def mc_rainbow_greeks(S0, K, T, r, sigmas, corr_matrix, n_paths=1_000_000, bump=1.0):
    # finite difference greeks via bump-and-reprice
    n_assets = len(S0)
    price_base, _, _, _ = mc_rainbow(S0, K, T, r, sigmas, corr_matrix, n_paths)

    deltas = np.zeros(n_assets)
    gammas = np.zeros(n_assets)

    for i in range(n_assets):
        S_up = S0.copy(); S_up[i] += bump
        S_dn = S0.copy(); S_dn[i] -= bump

        p_up, _, _, _ = mc_rainbow(S_up, K, T, r, sigmas, corr_matrix, n_paths)
        p_dn, _, _, _ = mc_rainbow(S_dn, K, T, r, sigmas, corr_matrix, n_paths)

        deltas[i] = (p_up - p_dn) / (2 * bump)
        gammas[i] = (p_up - 2 * price_base + p_dn) / (bump**2)

    return price_base, deltas, gammas


def build_corr_matrix(n_assets, rho=0.3):  # uniform pairwise correlation
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return corr


if __name__ == '__main__':
    # 2-asset case — compare against FDM
    print('=' * 50)
    print('2-ASSET RAINBOW — MC vs FDM comparison')
    print('=' * 50)
    S0_2 = np.array([100.0, 100.0])
    sigmas_2 = [0.2, 0.3]
    corr_2 = np.array([[1.0, 0.5], [0.5, 1.0]])

    t0 = time.time()
    price_2, stderr_2, _, _ = mc_rainbow(S0_2, 100, 1.0, 0.05, sigmas_2, corr_2, n_paths=10_000_000)
    t_2 = time.time() - t0
    print(f'MC price (10M paths): {price_2:.4f} ± {stderr_2:.4f}')
    print(f'FDM price:            18.7773 (from bs_fdm_2asset.py)')
    print(f'Difference:           {abs(price_2 - 18.7773):.4f}')
    print(f'Wall clock:           {t_2:.2f}s')

    # greeks for 2-asset
    print('\nGreeks (2-asset):')
    _, deltas_2, gammas_2 = mc_rainbow_greeks(S0_2, 100, 1.0, 0.05, sigmas_2, corr_2, n_paths=2_000_000)
    for i in range(2):
        print(f'  Asset {i+1}: Delta = {deltas_2[i]:.4f}, Gamma = {gammas_2[i]:.6f}')

    # 5-asset case — the actual target
    print('\n' + '=' * 50)
    print('5-ASSET RAINBOW — Ground Truth')
    print('=' * 50)
    S0_5 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    sigmas_5 = [0.2, 0.25, 0.3, 0.22, 0.28]
    corr_5 = build_corr_matrix(5, rho=0.3)

    t0 = time.time()
    price_5, stderr_5, S_T_5, payoff_5 = mc_rainbow(S0_5, 100, 1.0, 0.05, sigmas_5, corr_5, n_paths=10_000_000)
    t_5 = time.time() - t0
    print(f'MC price (10M paths): {price_5:.4f} ± {stderr_5:.4f}')
    print(f'Wall clock:           {t_5:.2f}s')

    # greeks for 5-asset
    print('\nGreeks (5-asset):')
    _, deltas_5, gammas_5 = mc_rainbow_greeks(S0_5, 100, 1.0, 0.05, sigmas_5, corr_5, n_paths=2_000_000)
    for i in range(5):
        print(f'  Asset {i+1} (σ={sigmas_5[i]}): Delta = {deltas_5[i]:.4f}, Gamma = {gammas_5[i]:.6f}')

    # ITM/OTM analysis
    itm_mask = payoff_5 > 0
    print(f'\nITM probability: {np.mean(itm_mask):.4f}')
    print(f'Mean payoff (ITM only): {np.mean(payoff_5[itm_mask]):.4f}')

    # plot payoff distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(payoff_5[payoff_5 > 0], bins=100, density=True, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Payoff')
    axes[0].set_ylabel('Density')
    axes[0].set_title('5-Asset Rainbow: Payoff Distribution (ITM only)')
    axes[0].axvline(price_5 * np.exp(0.05), color='red', linestyle='--', label=f'Undiscounted price')
    axes[0].legend()

    # convergence plot
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    prices_conv = []
    for n in path_counts:
        p, _, _, _ = mc_rainbow(S0_5, 100, 1.0, 0.05, sigmas_5, corr_5, n_paths=n)
        prices_conv.append(p)

    axes[1].semilogx(path_counts, prices_conv, 'o-', linewidth=2)
    axes[1].axhline(price_5, color='red', linestyle='--', alpha=0.5, label=f'10M estimate: {price_5:.4f}')
    axes[1].set_xlabel('Number of Paths')
    axes[1].set_ylabel('Price')
    axes[1].set_title('MC Convergence (5-Asset Rainbow)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mc_rainbow_analysis.png', dpi=150)
    print('\nSaved: mc_rainbow_analysis.png')
