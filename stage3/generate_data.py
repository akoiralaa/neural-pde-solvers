# generate Heston training data for DeepONet
# 1000 parameter sets → implied vol surfaces via characteristic function
# vectorized pricing for speed

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import time


def heston_price_vectorized(S0, K, T, r, kappa, theta, sigma, rho, v0, N_int=500):
    """Heston call price via char fn — vectorized over integration points."""
    ln_K = np.log(K)
    ln_S0 = np.log(S0)

    u = np.linspace(0.01, 100.0, N_int)  # integration grid

    # characteristic function evaluated at u and u-i
    def phi(u_arr):
        iu = 1j * u_arr
        d = np.sqrt((rho * sigma * iu - kappa)**2 + sigma**2 * (iu + u_arr**2))
        g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)

        exp_dT = np.exp(-d * T)
        C = (r * iu * T +
             (kappa * theta / sigma**2) *
             ((kappa - rho * sigma * iu - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))))
        D = ((kappa - rho * sigma * iu - d) / sigma**2) * ((1 - exp_dT) / (1 - g * exp_dT))

        return np.exp(C + D * v0 + iu * ln_S0)

    # P1: stock measure
    phi_u_mi = phi(u - 1j)
    phi_mi = phi(np.array([-1j]))[0]
    integrand1 = np.real(np.exp(-1j * u * ln_K) * phi_u_mi / (phi_mi * 1j * u))

    # P2: risk-neutral
    phi_u = phi(u)
    integrand2 = np.real(np.exp(-1j * u * ln_K) * phi_u / (1j * u))

    du = u[1] - u[0]
    P1 = 0.5 + np.trapezoid(integrand1, dx=du) / np.pi
    P2 = 0.5 + np.trapezoid(integrand2, dx=du) / np.pi

    return S0 * P1 - K * np.exp(-r * T) * P2


def bs_call(S, K, T, r, sigma):
    if T < 1e-10 or sigma < 1e-10:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return brentq(lambda sig: bs_call(S, K, T, r, sig) - price, 0.01, 3.0, xtol=1e-8)
    except ValueError:
        return np.nan


def sample_heston_params(n, rng):
    """Sample Heston params from tighter, more realistic ranges.

    Key constraint: soft Feller condition 2κθ > σ² reduces failures.
    """
    params = []
    while len(params) < n:
        kappa = rng.uniform(0.5, 4.0)
        theta = rng.uniform(0.02, 0.10)
        sigma = rng.uniform(0.1, 0.6)
        rho = rng.uniform(-0.9, -0.2)
        v0 = rng.uniform(0.01, 0.12)

        # soft Feller: 2κθ > 0.5*σ² (relaxed, allows some violation)
        if 2 * kappa * theta > 0.5 * sigma**2:
            params.append([kappa, theta, sigma, rho, v0])

    return np.array(params)


def generate_dataset(n_params=1000, seed=42):
    rng = np.random.default_rng(seed)
    S0 = 100.0
    r = 0.05

    moneyness = np.array([0.80, 0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15, 1.20])
    maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    strikes = S0 * moneyness

    n_K, n_T = len(strikes), len(maturities)
    n_points = n_K * n_T

    params = sample_heston_params(n_params, rng)

    all_params = []
    all_inputs = []
    all_ivs = []

    t0 = time.time()
    n_failed = 0

    for idx in range(n_params):
        kappa, theta, sigma, rho, v0 = params[idx]
        surface = np.zeros((n_T, n_K))
        valid = True

        for j, T in enumerate(maturities):
            for i, K in enumerate(strikes):
                try:
                    price = heston_price_vectorized(S0, K, T, r, kappa, theta, sigma, rho, v0)
                    if price < 0 or not np.isfinite(price):
                        valid = False; break
                    iv = implied_vol(price, S0, K, T, r)
                    if np.isnan(iv) or iv < 0.01 or iv > 2.0:
                        valid = False; break
                    surface[j, i] = iv
                except:
                    valid = False; break
            if not valid:
                break

        if not valid:
            n_failed += 1
            continue

        all_params.append(params[idx])
        for j, T in enumerate(maturities):
            for i, m in enumerate(moneyness):
                all_inputs.append([m, T])
                all_ivs.append(surface[j, i])

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            n_valid = len(all_params)
            print(f'  {idx+1}/{n_params} | {n_valid} valid | {n_failed} failed | '
                  f'{(idx+1)/elapsed:.1f} params/s')

    elapsed = time.time() - t0
    n_valid = len(all_params)

    all_params = np.array(all_params)
    all_inputs = np.array(all_inputs)
    all_ivs = np.array(all_ivs)

    print(f'\nDone: {n_valid} valid surfaces, {n_failed} failed, {elapsed:.1f}s')
    print(f'Shapes: params={all_params.shape}, inputs={all_inputs.shape}, ivs={all_ivs.shape}')

    np.savez('heston_data.npz',
             params=all_params,
             moneyness=moneyness,
             maturities=maturities,
             inputs=all_inputs,
             ivs=all_ivs,
             n_points_per_surface=n_points)

    print('Saved: heston_data.npz')
    return all_params, all_inputs, all_ivs


if __name__ == '__main__':
    print('Generating Heston implied vol dataset...\n')
    params, inputs, ivs = generate_dataset(n_params=2000)

    print(f'\nIV range: [{ivs.min():.4f}, {ivs.max():.4f}]')
    print(f'IV mean:  {ivs.mean():.4f}')
    print(f'IV std:   {ivs.std():.4f}')

    # show first surface
    n_pts = 11 * 7
    print(f'\nSample surface #1:')
    print(f'  κ={params[0,0]:.3f}, θ={params[0,1]:.4f}, σ={params[0,2]:.3f}, '
          f'ρ={params[0,3]:.3f}, v0={params[0,4]:.4f}')
    surface = ivs[:n_pts].reshape(7, 11)
    moneyness = np.array([0.80, 0.85, 0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15, 1.20])
    maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    print(f'  {"T\\m":>6}', ''.join(f'{m:>7.2f}' for m in moneyness))
    for j, T in enumerate(maturities):
        print(f'  {T:>6.2f}', ''.join(f'{surface[j,i]:>7.4f}' for i in range(11)))
