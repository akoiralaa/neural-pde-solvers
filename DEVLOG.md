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

### Laplace eigenfunctions on fractal domains — Koch snowflake training instability

Extended Stage 1 to solve ∇²u = -λu on complex 2D domains (Koch snowflake, five-pointed star). The eigenvalue λ is a free parameter learned during training (same idea as Y0 in the BSDE).

**Circle validation:** λ = 5.787 vs exact 5.783 (0.06% error). Confirms the architecture works.

**Koch snowflake problem:** Training was noisy — physics loss oscillates wildly between 0.03 and 300+. The fractal boundary has 768 edges (level 4) with tiny inlets that the PINN undersamps. The eigenfunction concentrates in the center and doesn't resolve boundary features.

**First fix attempt — multi-scale correction networks:** Train a base eigenfunction, then add correction networks focused on bump regions at each fractal level. Failed — corrections added noise instead of fixing the boundary. The correction networks were fighting the full residual field instead of just fixing local boundary errors. Residual actually got 0.3x worse.

**What worked — adaptive collocation resampling:** After initial training, compute the residual |∇²u + λu| everywhere. Resample collocation points weighted by residual² — this concentrates new points where the error is highest (near fractal inlets). Retrain with these focused points. Repeat for 4 phases.

**Result:** 2x residual reduction. The adaptive version found λ ≈ 23.9 vs single-scale λ ≈ 19.0. The residual heatmaps show less error concentration near the fractal boundary with adaptive sampling.

The key insight: it's better to retrain one network with better data than to stack correction networks. The corrections create coupling problems (each correction changes the Laplacian of the total, which invalidates the previous corrections). Adaptive resampling is simpler and more robust.

**Second fix attempt — curriculum training (level 2→3→4):** The hypothesis was that training on coarser Koch fractal levels first (fewer edges, smoother boundary) would give the network a good initialization before refining to level 4. Three changes: (1) curriculum on fractal level, (2) gradual point injection (replace 10% every 2000 epochs), (3) per-point loss normalization by nearest edge length.

**Result:** Only 1.1x improvement over baseline — essentially no gain. The curriculum approach failed for two reasons:

1. **Self-similarity kills transfer.** The Koch snowflake is self-similar: each level has the same fractal structure at a different scale. The eigenfunction on level 2 (48 edges) is genuinely different from level 4 (768 edges). The level-2 eigenvalue (λ ≈ 18.7) is far from the level-4 value (λ ≈ 20.6), so the "warm start" puts the network in a different basin, not a better one.

2. **Point injection causes the spikes, not the fractal boundary.** The physics loss spikes to 1000-5000 every ~2000 epochs — exactly when the gradual injection replaces 10% of collocation points. Suddenly replacing training data mid-optimization creates a temporary distribution shift that the Adam optimizer needs several hundred steps to recover from. The spikes happen in both the curriculum and baseline (which also uses adaptive resampling).

**Takeaway:** For fractal PINNs, adaptive resampling (2x improvement) is the practical tool. Curriculum on fractal level doesn't transfer because the eigenfunction structure changes at each level. The remaining oscillation is caused by collocation point replacement — this is an inherent limitation of adaptive resampling with a finite point budget.

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

**Correlation mismatch bug:** The 2-asset scaling test failed at 5.35% for two runs in a row, with and without residual connections. Both converged to Y0 ≈ 20.88 vs MC price 18.84. The real bug: the MC reference price (18.8423) was computed with ρ=0.5 between the two assets (matching `mc_rainbow.py`), but `deep_bsde_fix.py` used ρ=0.3 for all dimensions. Lower correlation → higher max(S1,S2) → higher option price. The BSDE was converging correctly for its inputs — just to a different problem than the reference. Fixed by using ρ=0.5 for the 2-asset case to match the MC setup.

---

## Stage 3 — Heston DeepONet

### Heston FD solver — explicit scheme NaN blowup (again)

First attempt used explicit Euler on the (S, v) grid. Same failure as Stage 2: the 0.5·v·S² coefficient in the diffusion term grows without bound in the upper-right corner of the grid. CFL violated, NaN within a few timesteps.

**Fix:** Two changes. (1) Log-spot transform x = ln(S) removes the S² growth — all coefficients in the transformed PDE are bounded. (2) Fully implicit time-stepping. Built the sparse matrix once (interior coefficients are time-independent), update only the boundary RHS each step. Went from 29s to 11s per solve.

**Result:** ATM price 10.3665 vs semi-analytical 10.3140, error 0.51%. Good enough for validation. Used the char fn (12ms, more accurate) for data generation instead.

---

### Data generation — 57% failure rate on first attempt

First parameter sampling used wide ranges (σ up to 0.8, no Feller filter). 867/1000 failed — the characteristic function produces negative prices or the IV inversion fails for extreme parameters.

**Root cause:** Two issues. (1) High vol-of-vol (σ > 0.6) combined with low mean reversion makes the char fn numerically unstable — the complex exponentials overflow. (2) Deep OTM options at short maturities have prices near zero, making IV inversion degenerate.

**Fix:** Tightened ranges (σ ≤ 0.6, θ ≤ 0.10) and added soft Feller filter (2κθ > 0.5σ²). Also vectorized the char fn with numpy — 50x speedup over the cmath loop version. Generated 2000 candidates to get 874 valid surfaces in 40 seconds.

---

### numpy 2.x breaking change — np.trapz removed

`np.trapz` was removed in NumPy 2.x (Python 3.14 ships with a recent numpy). Silent `AttributeError` that looked like a deeper bug until isolated.

**Fix:** Replace with `np.trapezoid`.

---

### DeepONet — worked first try

Surprising. The branch/trunk architecture with 256-hidden, 128-latent, 3 layers each converged to 0.38% mean relative error on 175 test surfaces in 54 seconds. No tuning needed beyond standard Adam + cosine annealing.

The key design decisions that probably helped:
1. Input normalization (standardize both branch and trunk inputs)
2. Predicting implied vol instead of price — vol surfaces are smoother and more bounded than price surfaces
3. Latent dimension 128 gives enough capacity for the 5→surface mapping

**Failure mode confirmed:** 4/175 test surfaces exceeded 1% error, all near the Feller boundary (2κθ/σ² ≈ 1) or with weak leverage (ρ near -0.2). The README predicted "the network will fail on extreme regimes" — this is exactly what happened.

---

## Infrastructure

### matplotlib savefig path error

The PINN script ran from `stage1/` but savefig paths were `stage1/pinn_comparison.png` (relative to project root). FileNotFoundError because `stage1/stage1/` doesn't exist.

**Fix:** Changed save paths to just filenames since the script runs from within the stage directory. Simple but easy to miss when you have scripts that might run from different working directories.

### Python 3.14 + PyTorch compatibility

Running Python 3.14.2 (bleeding edge). PyTorch 2.10.0 installed cleanly via pip on Apple Silicon (MPS backend). No compatibility issues hit so far, but worth noting since 3.14 is not yet widely tested.
