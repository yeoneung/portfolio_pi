"""Regenerate benchmark slice and training figures from supplied data
for the N_30_M10 benchmark cell.

Sources (inside code/pipinn_nemerical):
  outputs/pipinn/N_30_M10/publication_slice.npz   -> 9 slice heatmaps
  outputs/pipinn/N_30_M10/outer_history.csv       -> policy_convergence.png
  outputs/pinn/N_30_M10/train_history.csv         \
  comparisons/training_curves/N_30_M10/...json    -> train_curve.png

Conventions (gamma = 2, x = 1):
  value  V = x^(1-gamma)/(1-gamma) * g = -g
  exposure theta/x = vartheta, plotted for representative asset index 16.

Outputs land in this folder, with provenance.json recording the source hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "code" / "pipinn_nemerical" / "outputs"
CELL = OUT_ROOT / "outputs" / "pipinn" / "N_30_M10"
PINN_CELL = OUT_ROOT / "outputs" / "pinn" / "N_30_M10"
CMP = OUT_ROOT / "comparisons" / "training_curves" / "N_30_M10"

GAMMA = 2.0
ASSET = 16
DPI = 300


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def heat(fig_name, tau, z, values, cbar_label, diverging=False):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    if diverging:
        vmax = float(np.max(np.abs(values)))
        mesh = ax.pcolormesh(z, tau, values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="gouraud")
    else:
        mesh = ax.pcolormesh(z, tau, values, cmap="viridis", shading="gouraud")
    ax.set_xlabel(r"state $z_1$")
    ax.set_ylabel(r"$\tau$")
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(HERE / fig_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fig_name)


def main():
    slice_npz = CELL / "publication_slice.npz"
    d = np.load(slice_npz)
    tau, z = d["tau_grid"], d["x_grid"]

    V_pred, V_exact = -d["g_pred"], -d["g_exact"]
    heat("pipinn_value.png", tau, z, V_pred, r"$V(1,z)$")
    heat("pipinn_value_closed_form.png", tau, z, V_exact, r"$V(1,z)$")
    heat("pipinn_value_diff.png", tau, z, V_pred - V_exact, "error", diverging=True)

    th_p, th_e = d["vartheta_pred"][..., ASSET], d["vartheta_exact"][..., ASSET]
    heat("pipinn_theta.png", tau, z, th_p, rf"$\vartheta_{{{ASSET + 1}}}$")
    heat("pipinn_theta_closed_form.png", tau, z, th_e, rf"$\vartheta_{{{ASSET + 1}}}$")
    heat("pipinn_theta_diff.png", tau, z, th_p - th_e, "error", diverging=True)

    hg_p, hg_e = d["hedging_pred"][..., ASSET], d["hedging_exact"][..., ASSET]
    heat("pipinn_hedging.png", tau, z, hg_p, rf"$\vartheta^{{\rm hed}}_{{{ASSET + 1}}}$")
    heat("pipinn_hedging_closed_form.png", tau, z, hg_e,
         rf"$\vartheta^{{\rm hed}}_{{{ASSET + 1}}}$")
    heat("pipinn_hedging_diff.png", tau, z, hg_p - hg_e, "error", diverging=True)

    # ---- training curves ---------------------------------------------------
    cmp_json = json.loads((CMP / "training_loss_pinn_vs_pipinn.json").read_text())
    pinn_series = np.asarray(
        [p["total_training_loss"] for p in cmp_json["pinn"]["series"]], dtype=float)
    pipinn_series = np.asarray(
        [p["accepted_train_total"] for p in cmp_json["pipinn"]["series"]], dtype=float)
    outer = np.arange(1, len(pinn_series) + 1)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.semilogy(outer, pinn_series, "o-", label="PINN (nonlinear residual)")
    ax.semilogy(outer, pipinn_series, "s-", label="PINN-PI (frozen linear residual)")
    ax.set_xlabel("outer iteration / resampling block")
    ax.set_ylabel("training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "train_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote train_curve.png")

    with open(CELL / "outer_history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    theta_diff = np.asarray([float(r["theta_diff_mse"]) for r in rows])
    outer2 = np.arange(1, len(theta_diff) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.semilogy(outer2, theta_diff, "s-")
    ax.set_xlabel("outer iteration $n$")
    ax.set_ylabel(r"$\|\theta_{n+1}/x-\theta_n/x\|_2^2$ (mean)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "policy_convergence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote policy_convergence.png")

    provenance = {
        "cell": "N_30_M10 (n_assets=30, m_states=10, market_seed=12)",
        "asset_index_plotted": ASSET,
        "value_convention": "V = x^{1-gamma}/(1-gamma) g at x=1 (= -g for gamma=2)",
        "sources": {
            "publication_slice.npz": sha256(slice_npz),
            "outer_history.csv": sha256(CELL / "outer_history.csv"),
            "training_loss_pinn_vs_pipinn.json":
                sha256(CMP / "training_loss_pinn_vs_pipinn.json"),
        },
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("wrote provenance.json")


if __name__ == "__main__":
    main()
