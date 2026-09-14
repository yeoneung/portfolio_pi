"""Reproduce analytic localization budgets; no PDE solutions are inferred."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import portfolio_welfare as w
from direct_residual_certificate import upper_sci


def main():
    rows = []
    for horizon in (F(1, 8), F(1, 4), F(1, 2), F(1)):
        for radius in (F(1), F(3, 2), F(2), F(5, 2)):
            market = replace(w.Market(), radius=radius, horizon=horizon)
            market.validate()
            w.trace_slopes(market)
            result = w.localization(market)
            rows.append({"horizon": str(horizon), "radius": str(radius), **result})
    output = ROOT / "results" / "portfolio_welfare" / "localization_budget.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["% Analytic localization upper bounds from analysis/localization_budget.py.",
             r"\begin{tabular}{lrrrr}\toprule",
             r"Horizon & $R=1$ & $R=1.5$ & $R=2$ & $R=2.5$\\\midrule"]
    for horizon in (F(1, 8), F(1, 4), F(1, 2), F(1)):
        values = [r for r in rows if r["horizon"] == str(horizon)]
        table.append("$"+str(horizon)+"$ & "+" & ".join("$"+upper_sci(v["value_localization_upper"])+"$" for v in values)+r"\\")
    table.append(r"\bottomrule\end{tabular}")
    table_path = ROOT / "results" / "tables" / "tab_localization_budget.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(table)+"\n")
    print(f"Generated {len(rows)} analytic budget entries: {output}")


if __name__ == "__main__":
    main()
