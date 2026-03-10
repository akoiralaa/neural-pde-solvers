# 1D heat equation PINN — soft BC vs hard BC (ansatz) comparison
# trains both variants, compares against exact fourier solution

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from heat_equation_exact import exact_solution_single_mode

ALPHA = 0.01  # thermal diffusivity
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PINN(nn.Module):  # simple MLP — 4 hidden layers, 40 neurons, tanh
    def __init__(self, hard_bc=False):
        super().__init__()
        self.hard_bc = hard_bc
        self.net = nn.Sequential(
            nn.Linear(2, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1)
        )
        self._init_weights()

    def _init_weights(self):  # xavier init for tanh
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        u = self.net(inputs)
        if self.hard_bc:  # ansatz: u = x*(1-x)*NN, guarantees u(0,t)=u(1,t)=0
            u = x * (1.0 - x) * u
        return u


def physics_residual(model, x, t, alpha=ALPHA):  # r = du/dt - alpha*d²u/dx²
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]

    return du_dt - alpha * d2u_dx2  # should be 0 if PDE is satisfied


def sample_points(n_interior, n_bc, n_ic):  # random collocation points
    # interior: random (x, t) in (0,1) x (0,1)
    x_int = torch.rand(n_interior, 1)
    t_int = torch.rand(n_interior, 1)

    # boundary: x=0 and x=1, random t
    t_bc = torch.rand(n_bc, 1)
    x_bc_left = torch.zeros(n_bc, 1)
    x_bc_right = torch.ones(n_bc, 1)

    # initial condition: random x, t=0
    x_ic = torch.rand(n_ic, 1)
    t_ic = torch.zeros(n_ic, 1)
    u_ic = torch.sin(np.pi * x_ic)  # sin(pi*x)

    return (x_int.to(DEVICE), t_int.to(DEVICE),
            x_bc_left.to(DEVICE), x_bc_right.to(DEVICE), t_bc.to(DEVICE),
            x_ic.to(DEVICE), t_ic.to(DEVICE), u_ic.to(DEVICE))


def train(model, epochs=15000, lr=1e-3, n_interior=2000, n_bc=200, n_ic=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    history = {"total": [], "physics": [], "bc": [], "ic": []}

    for epoch in range(1, epochs + 1):
        x_int, t_int, x_bl, x_br, t_bc, x_ic, t_ic, u_ic = sample_points(n_interior, n_bc, n_ic)

        # physics loss
        r = physics_residual(model, x_int, t_int)
        loss_physics = torch.mean(r ** 2)

        # boundary loss
        u_left = model(x_bl, t_bc)
        u_right = model(x_br, t_bc)
        loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

        # initial condition loss
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

        loss = loss_physics + 10.0 * loss_bc + 10.0 * loss_ic  # weight BC/IC higher

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(loss.item())
        history["physics"].append(loss_physics.item())
        history["bc"].append(loss_bc.item())
        history["ic"].append(loss_ic.item())

        if epoch % 1000 == 0 or epoch == 1:
            print(f"epoch {epoch:5d} | total: {loss.item():.6f} | "
                  f"physics: {loss_physics.item():.6f} | bc: {loss_bc.item():.6f} | ic: {loss_ic.item():.6f}")

    return history


def evaluate(model, nx=100, nt=100):  # compare against exact on a grid
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(nt, nx)

    u_exact = exact_solution_single_mode(X, T, ALPHA)
    error = np.abs(u_pred - u_exact)

    return X, T, u_pred, u_exact, error


def plot_comparison(results_soft, results_hard):
    X, T, u_soft, u_exact, err_soft = results_soft
    _, _, u_hard, _, err_hard = results_hard

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # row 1: soft BC
    axes[0, 0].pcolormesh(X, T, u_exact, shading="auto", cmap="inferno")
    axes[0, 0].set_title("Exact Solution")
    axes[0, 1].pcolormesh(X, T, u_soft, shading="auto", cmap="inferno")
    axes[0, 1].set_title("PINN — Soft BC")
    im1 = axes[0, 2].pcolormesh(X, T, err_soft, shading="auto", cmap="hot")
    axes[0, 2].set_title(f"Error (Soft BC) — L∞ = {np.max(err_soft):.2e}")
    plt.colorbar(im1, ax=axes[0, 2])

    # row 2: hard BC (ansatz)
    axes[1, 0].pcolormesh(X, T, u_exact, shading="auto", cmap="inferno")
    axes[1, 0].set_title("Exact Solution")
    axes[1, 1].pcolormesh(X, T, u_hard, shading="auto", cmap="inferno")
    axes[1, 1].set_title("PINN — Hard BC (Ansatz)")
    im2 = axes[1, 2].pcolormesh(X, T, err_hard, shading="auto", cmap="hot")
    axes[1, 2].set_title(f"Error (Hard BC) — L∞ = {np.max(err_hard):.2e}")
    plt.colorbar(im2, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    plt.tight_layout()
    plt.savefig("pinn_comparison.png", dpi=150)
    print("Saved: pinn_comparison.png")


def plot_boundary_error(results_soft, results_hard):  # the key comparison
    X, T, _, _, err_soft = results_soft
    _, _, _, _, err_hard = results_hard

    # error at x=0 boundary over time
    t_vals = T[:, 0]
    bc_err_soft = err_soft[:, 0]  # x=0 column
    bc_err_hard = err_hard[:, 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t_vals, bc_err_soft, label="Soft BC", linewidth=2)
    ax.semilogy(t_vals, bc_err_hard, label="Hard BC (Ansatz)", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("Absolute Error at x = 0")
    ax.set_title("Boundary Error: Soft vs Hard BC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("boundary_error.png", dpi=150)
    print("Saved: boundary_error.png")


def plot_loss_curves(hist_soft, hist_hard):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, hist, title in [(axes[0], hist_soft, "Soft BC"), (axes[1], hist_hard, "Hard BC (Ansatz)")]:
        ax.semilogy(hist["total"], label="Total", alpha=0.8)
        ax.semilogy(hist["physics"], label="Physics", alpha=0.8)
        ax.semilogy(hist["bc"], label="BC", alpha=0.8)
        ax.semilogy(hist["ic"], label="IC", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    print("Saved: loss_curves.png")


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    # train soft BC model
    print("=" * 50)
    print("Training PINN — Soft BC")
    print("=" * 50)
    model_soft = PINN(hard_bc=False).to(DEVICE)
    hist_soft = train(model_soft)

    # train hard BC model (ansatz)
    print("\n" + "=" * 50)
    print("Training PINN — Hard BC (Ansatz)")
    print("=" * 50)
    model_hard = PINN(hard_bc=True).to(DEVICE)
    hist_hard = train(model_hard)

    # evaluate both
    results_soft = evaluate(model_soft)
    results_hard = evaluate(model_hard)

    # summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Soft BC  — L∞ error: {np.max(results_soft[4]):.6e}")
    print(f"Hard BC  — L∞ error: {np.max(results_hard[4]):.6e}")
    print(f"Soft BC  — boundary error (x=0): {np.max(results_soft[4][:, 0]):.6e}")
    print(f"Hard BC  — boundary error (x=0): {np.max(results_hard[4][:, 0]):.6e}")

    # plots
    plot_comparison(results_soft, results_hard)
    plot_boundary_error(results_soft, results_hard)
    plot_loss_curves(hist_soft, hist_hard)

    # save models
    torch.save(model_soft.state_dict(), "model_soft.pt")
    torch.save(model_hard.state_dict(), "model_hard.pt")
    print("\nModels saved.")
