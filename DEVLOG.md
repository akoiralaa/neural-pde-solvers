# Development Log

Documenting every problem hit during development and how it was solved. Not cleaned up for presentation — this is the raw engineering record.

---

## Stage 1 — 1D Heat Equation PINN

### Fourier series validator off by O(1)

The general Fourier series function was returning wildly wrong results when compared against the known exact solution for sin(πx) initial condition. Error was ~0.95, meaning the solution was basically wrong everywhere.

**Root cause:** The function had an `ic=None` fallback path that returned a 1D array, but the caller expected a 2D array (time x space). When `ic` was not passed explicitly, it silently fell through to the wrong code path and shapes didn't match during the comparison.

**Fix:** Always pass the IC function explicitly when validating. The `None` default was a convenience shortcut that hid a shape mismatch bug. Caught by assertion, not by visual inspection — which is exactly why assertions exist.

**Takeaway:** Default arguments that change return shapes are a bad pattern. Would have been invisible without the explicit numerical check.

---

### Soft BC PINN — boundary error floor at ~10⁻³

The soft BC model trained cleanly (loss dropped to ~10⁻⁴) but boundary error at x=0 and x=1 never got below ~10⁻³. This is the expected behavior, not a bug — the BC is enforced as a penalty term, so the optimizer balances it against the physics loss. It can get close to zero but never exactly zero.

Increasing the BC weight from 10 to 100 pushes the boundary error lower but hurts the interior accuracy. There's no free lunch.

**Resolution:** This is exactly what Stage 1 was designed to demonstrate. The hard BC (ansatz) model achieves exactly zero boundary error by construction: `u = x(1-x)*NN(x,t)`. No tuning needed. The ansatz removes the boundary from the optimization problem entirely.

**Result:** Hard BC L∞ = 5.97e-04, Soft BC L∞ = 9.15e-03. Boundary error: hard = 0.0, soft = 2.54e-03. The ansatz wins by 15x overall and infinitely at the boundary.

---

## Stage 2 — Multi-Asset Rainbow Option

### Crank-Nicolson NaN blowup (2-asset FDM)

First implementation of the 2D Black-Scholes FDM used Crank-Nicolson (half explicit, half implicit). Produced NaN at ATM price on a 80x80 grid with 200 time steps.

**Root cause:** The cross-derivative term (ρ·σ₁·σ₂·S₁·S₂ · ∂²V/∂S₁∂S₂) was split incorrectly between the explicit and implicit sides. The explicit contribution was destabilizing the scheme because the cross term coefficients grow with S₁·S₂, meaning the effective CFL condition was violated in the upper-right region of the grid where both assets are large.

Mixed explicit-implicit handling of the cross term is a known instability source in multi-asset FD schemes. The safe options are: fully implicit (stable but slower per step), ADI splitting (splits dimensions but requires careful operator ordering), or just using more time steps to satisfy CFL.

**Fix:** Switched to fully implicit Euler. All spatial terms are on the implicit side, solved via sparse linear system at each timestep. Unconditionally stable — no CFL constraint. Slightly more diffusive than Crank-Nicolson but for a validation baseline, stability matters more than sharpness.

**Result:** ATM price V(100,100) = 18.7773 on a 60x60 grid, 300 time steps, 3.4 seconds. Matches Monte Carlo (18.8245 ± 0.007) to within grid error.

---

### Deep BSDE — Y0 not converging (stuck at ~2.5 instead of ~18.8)

First Deep BSDE implementation: Y0 initialized at 1.0, 30 time steps, batch 4096, 3000 epochs Adam at lr=1e-3. Loss starts at 114k and drops to ~300 but Y0 only reaches ~2.7. Predicted price is off by 84%.

**What was tried:**
1. BatchNorm on subnet inputs — helped stability but Y0 still stuck
2. Input normalization (X/100) — marginal improvement
3. Cosine annealing scheduler — no meaningful change
4. Larger networks (128 → 256 hidden) — no change
5. Better Y0 initialization (starting at 15.0 instead of 1.0) — Y0 drifts back down

**Diagnosis in progress:** The loss of ~300 means E[(Y_T - g(X_T))²] ≈ 300, so the average error in terminal matching is ~√300 ≈ 17. The payoffs range from 0 to ~200. The Z networks are not learning the gradient of the solution correctly — the stochastic integral Σ Z·σ·X·dW is not accumulating to the right total over 24-30 time steps.

Likely issues:
- The rainbow `max` payoff creates a non-smooth terminal condition. The gradient ∇u has discontinuities at S_i = S_j surfaces, which is hard for the subnet to learn.
- The product Z·σ·X·dW has terms of order Z·0.2·100·√(1/24) ≈ 4·Z. Over 24 steps, total contribution ≈ 24·4·Z. For Y to go from 15 to ~30 (terminal), Z needs to be of a specific magnitude and sign pattern that depends on the path. The network might need more capacity or a different parameterization.

**Debugging approach:** Isolated the problem by testing on a 1D European call first (see `deep_bsde_debug.py`). The 1D call converged to within 1.58% of Black-Scholes analytical price. This confirmed the BSDE formulation was correct — the issue was specific to how the multi-asset version was parameterized, not a fundamental architecture bug.

**What actually fixed it:**
1. Initializing Y0 near the expected price (15.0 for 2-asset, 25.0 for 5-asset) instead of 1.0. The optimizer was wasting thousands of epochs just moving Y0 from 1 to the right neighborhood.
2. Higher learning rate (5e-3 vs 1e-3) — the problem needs aggressive early learning to get Y0 into the right range.
3. Simpler subnet architecture (64 hidden, no BatchNorm) — the bigger 256-hidden networks with BN were overfitting to noise in the stochastic gradients.
4. Normalizing subnet inputs by X/100 to keep them near unit scale.

**Final result:** 2-asset error 0.07%, 5-asset error 0.01%. Both well under the 0.5% target.

---

### Deep BSDE — Greeks are wrong even when price is correct

The predicted failure mode from the roadmap actually happened. Prices match MC to <0.1% but deltas are near zero instead of 0.19-0.29. The BSDE learns Y0 (the option price) as a free parameter but the Z networks (which encode the gradient ∇u) don't learn the correct spatial sensitivity. Bump-and-reprice on the BSDE model gives near-zero deltas because the model was only trained at one spot price — it doesn't know how the price changes when S0 moves.

This is a structural limitation of the Deep BSDE approach: it learns the price at a single initial condition, not a function over the domain. To get correct Greeks, you'd need to either train across multiple S0 values or use a different method entirely (like a PINN that learns the full solution surface).

---

### Deep BSDE — scaling failure at 50 and 100 assets

Initial scaling test used the same 64-hidden subnets that worked for 2 and 5 assets. Results at higher dimensions:

- 2-asset: 0.12% (pass)
- 5-asset: 0.12% (pass)
- 20-asset: 0.18% (pass)
- 50-asset: 4.73% (fail)
- 100-asset: 1.76% (fail)

**Root cause:** The Z subnets need to approximate ∇u ∈ ℝ^d. At d=50 or d=100, a 64-unit hidden layer doesn't have enough capacity to represent the gradient of the solution — the rainbow payoff max(S1,...,Sd) creates a piecewise-linear terminal condition with d faces, and the gradient has discontinuities along the d(d-1)/2 hyperplanes where S_i = S_j. More capacity is needed to approximate this in higher dimensions.

**Fix (deep_bsde_fix.py):**
1. Wider subnets: 256 hidden units (up from 64)
2. Residual connections: `return out + x` — each subnet learns a correction to the identity mapping instead of the full Z from scratch. This stabilizes training because the initial output is close to the input, not random noise.
3. More timesteps: N=40 (up from 20-30). Finer time discretization reduces the per-step approximation error in the Euler-Maruyama scheme.
4. Longer training: 15000 epochs with MultiStepLR (milestones at 4000, 8000, 12000, gamma=0.3)

**Results after fix:**
- 50-asset: 77.3570 vs MC 77.3326 → 0.03% (pass)
- 100-asset: 79.4686 vs MC 79.4396 → 0.04% (pass)
- 2-asset: 18.8281 vs MC 18.8423 → 0.08% (pass)

The residual connection was the key insight. Without it, the 256-hidden nets still struggled because the initial random weights produced large random Z outputs that destabilized the SDE simulation in early training.

---

## Infrastructure

### matplotlib savefig path error

The PINN script ran from `stage1/` but savefig paths were `stage1/pinn_comparison.png` (relative to project root). FileNotFoundError because `stage1/stage1/` doesn't exist.

**Fix:** Changed save paths to just filenames since the script runs from within the stage directory. Simple but easy to miss when you have scripts that might run from different working directories.

### Python 3.14 + PyTorch compatibility

Running Python 3.14.2 (bleeding edge). PyTorch 2.10.0 installed cleanly via pip on Apple Silicon (MPS backend). No compatibility issues hit so far, but worth noting since 3.14 is not yet widely tested.
