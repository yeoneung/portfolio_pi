"""Log-log Newton-rate figure from the exact-PI ODE experiment results."""

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exact_pi_ode" / "exact_pi_ode_results.json"

data = json.loads(RES.read_text())
fig, ax = plt.subplots(figsize=(4.6, 3.6))
markers = {"M1": "o", "M3": "s", "M5": "^", "M10": "d"}
for key, res in data.items():
    pairs = res["newton_fit_pairs"]
    e_n = np.array([p["e_n"] for p in pairs])
    e_n1 = np.array([p["e_n1"] for p in pairs])
    ax.loglog(e_n, e_n1, markers[key], ms=5, ls="none",
              label=rf"$L={key[1:]}$ (slope {res['newton_loglog_slope']:.2f})")

grid = np.logspace(-4.3, 0.2, 50)
ax.loglog(grid, grid ** 2, "k--", lw=1, label=r"slope $2$ reference")
ax.set_xlabel(r"$e_n$")
ax.set_ylabel(r"$e_{n+1}$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(HERE / "newton_rate.png", dpi=300, bbox_inches="tight")
print("wrote newton_rate.png")
