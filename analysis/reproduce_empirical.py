"""Reproduce the empirical tables from supplied monthly strategy paths.

No retraining or network access is performed.
RF is read from the checksum-verified July-2025 Ken French source ZIP.
The statistical calculation conditions on the selected specifications.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from empirical_statistics import (
    _total_metrics, _circular_block_bootstrap, _circular_block_bootstrap_total,
    _hac_total_rows, _crra_break_even_cost,
)

ROOT = Path(__file__).resolve().parents[1]
RF_SHA256 = "42492fc7fe23de2c44058e77414e1d35c451bf8d289dfff75c5b80c2804324e2"
UNIVERSES = ("ff10", "ff12", "ff17", "ff25")


def load_rf(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != RF_SHA256:
        raise ValueError("Risk-free source ZIP checksum mismatch")
    with zipfile.ZipFile(path) as source:
        members = [n for n in source.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one CSV in the RF source ZIP")
        lines = source.read(members[0]).decode("utf-8-sig").splitlines()
    rows = [row for row in csv.reader(lines) if row and re.fullmatch(r"\d{6}", row[0].strip())]
    dates = pd.to_datetime([r[0].strip() for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    rf = pd.Series([float(r[6]) / 100 for r in rows], index=dates, name="RF")
    if rf.index.duplicated().any() or not np.isfinite(rf.to_numpy()).all():
        raise ValueError("Invalid monthly RF data")
    return rf


def load_pair(bundle, universe):
    path = (bundle / "comparative" / f"{universe}_pls_tau3_fixed" /
            "selected_by_pipinn" / "rank_001" / "run_as_pipinn" / "comparison_results.csv")
    frame = pd.read_csv(path, parse_dates=["return_date"])
    pieces = []
    for strategy, mode in (("pipinn", "estimated"), ("myopic", "reference")):
        part = frame.loc[(frame.strategy == strategy) & (frame.cross_mode == mode),
                         ["return_date", "gross_return", "turnover"]].copy()
        if part.empty or part.return_date.duplicated().any():
            raise ValueError(f"Invalid {strategy} dates in {path}")
        pieces.append(part.set_index("return_date").sort_index().add_prefix(strategy + "_"))
    pair = pieces[0].join(pieces[1], how="inner")
    if len(pair) != len(pieces[0]) or len(pair) != len(pieces[1]):
        raise ValueError(f"Unaligned strategy dates in {path}")
    return pair, path


def summarize_pair(rows, universe, a, ea, ta, b, eb, tb, **fields):
    ma, mb = _total_metrics(a, ea, ta, gamma=5), _total_metrics(b, eb, tb, gamma=5)
    for strategy, values in (("pipinn", ma), ("myopic", mb),
                             ("pipinn_minus_myopic", {k: ma[k] - mb[k] for k in ma})):
        rows.append({"universe": universe, **fields, "strategy": strategy, "months": len(a), **values})
    return ma, mb


def write_tables(out, frames, reps):
    """Generate the two empirical tables, without editing surrounding prose."""
    primary, boot, hac, costs, subperiod, breaks = frames
    lines = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{PINN-PI and the matched myopic benchmark. Ret., Vol., CRRA CER, and MDD use reconstructed total returns; Sharpe uses excess returns. PINN-PI enforces the long-only constraint at every improvement step. All rows are out of sample under the stated rolling protocol and before transaction costs.}",
             r"\label{tab:pinnpi-mv}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Universe & Strategy & Ret. & Vol. & Excess Sharpe & CRRA CER & Turnover & MDD\\",
             r"\midrule", r"\multicolumn{8}{l}{\emph{Out-of-sample 2000--2019, rolling 240-month windows, annual refits, monthly rebalancing}}\\",
             r"\addlinespace[2pt]"]
    metrics = ["ann_geometric_total_return", "ann_total_volatility", "excess_sharpe", "crra_total_cer", "avg_turnover", "max_drawdown_total"]
    for u in UNIVERSES:
        for strategy, name in (("myopic", "MV"), ("pipinn", "PINN-PI")):
            row = primary.loc[(primary.universe == u) & (primary.strategy == strategy)].iloc[0]
            lines.append(f"{u.upper() if strategy == 'myopic' else ''} & {name} & " +
                         " & ".join(f"{row[k]:.4f}" for k in metrics) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_pinnpi_mv.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{table}[tbp]", r"\centering\scriptsize",
             rf"\caption{{Robustness of the PINN-PI minus myopic CRRA-CER difference, using reconstructed total returns. The circular block bootstrap uses {reps:,} paired resamples with 12-month blocks; HAC intervals use 12 Newey--West lags. Intervals are specification-conditional and computed separately for each universe. Costs are charged on full risky-leg $L^1$ turnover with the zero-cost-trained policies held fixed.}}",
             r"\label{tab:emp-robustness}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Universe & 0 bp $\Delta$CER & Block 95\% CI & HAC 95\% CI & 10 bp & 25 bp & Break-even (bp)\\", r"\midrule"]
    for u in UNIVERSES:
        delta = primary.loc[(primary.universe == u) & (primary.strategy == "pipinn_minus_myopic"), "crra_total_cer"].iloc[0]
        ci = boot.loc[(boot.universe == u) & (boot.metric == "crra_total_cer")].iloc[0]
        hi = hac.loc[(hac.universe == u) & (hac.metric == "crra_total_cer")].iloc[0]
        cost = costs.loc[(costs.universe == u) & (costs.strategy == "pipinn_minus_myopic")].set_index("cost_bps")
        be = breaks.loc[breaks.universe == u, "break_even_bps"].iloc[0]
        be_text = f"{be:.1f}" if np.isfinite(be) else "---"
        lines.append(f"{u.upper()} & {delta:.4f} & [{ci.ci_low:.4f}, {ci.ci_high:.4f}] & "
                     f"[{hi.ci_low:.4f}, {hi.ci_high:.4f}] & {cost.loc[10, 'crra_total_cer']:.4f} & "
                     f"{cost.loc[25, 'crra_total_cer']:.4f} & {be_text}" + r"\\")
    lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{Subperiod $\Delta$CER (before transaction costs)}}\\",
              r"Universe & 2000--2009 & 2010--2019 & & & &\\"]
    for u in UNIVERSES:
        row = subperiod.loc[(subperiod.universe == u) & (subperiod.strategy == "pipinn_minus_myopic")].set_index("subperiod")
        lines.append(f"{u.upper()} & {row.loc['2000-2009', 'crra_total_cer']:.4f} & "
                     f"{row.loc['2010-2019', 'crra_total_cer']:.4f} & & & &" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_empirical_robustness.tex").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-reps", type=int, default=30000)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "empirical_tables")
    p.add_argument("--update-tables", action="store_true")
    args = p.parse_args(argv)
    if args.bootstrap_reps < 1:
        p.error("bootstrap-reps must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = ROOT / "code" / "empirical_src_bundle"
    rf_path = ROOT / "results" / "empirical_robustness" / "source_data" / "F-F_Research_Data_5_Factors_2x3_CSV_2025-07.zip"
    rf = load_rf(rf_path)
    rng = np.random.default_rng(20260907)
    primary, bootstrap, hac, costs, subperiods, breaks, sources = [], [], [], [], [], [], []
    for universe in UNIVERSES:
        pair, path = load_pair(bundle, universe)
        source_rf = rf.reindex(pair.index)
        if source_rf.isna().any():
            raise ValueError(f"Missing RF dates for {universe}")
        ea, eb = pair.pipinn_gross_return.to_numpy(), pair.myopic_gross_return.to_numpy()
        ta, tb = pair.pipinn_turnover.to_numpy(), pair.myopic_turnover.to_numpy()
        a, b = ea + source_rf.to_numpy(), eb + source_rf.to_numpy()
        ma, mb = summarize_pair(primary, universe, a, ea, ta, b, eb, tb)
        # A paired excess-statistic panel consumes the first block sample;
        # the total-return panel uses the second, with the same fixed seed.
        _circular_block_bootstrap(ea, eb, reps=args.bootstrap_reps, block_length=12, rng=rng, gamma=5)
        samples = _circular_block_bootstrap_total(a, ea, b, eb, reps=args.bootstrap_reps,
                                                 block_length=12, rng=rng, gamma=5)
        for metric, values in samples.items():
            bootstrap.append({"universe": universe, "metric": metric, "estimate": ma[metric] - mb[metric],
                              "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                              "bootstrap_prob_delta_gt_zero": float(np.mean(values > 0)),
                              "bootstrap_reps": args.bootstrap_reps, "block_length_months": 12, "seed": 20260907})
        hac.extend(_hac_total_rows(universe, a, ea, b, eb, gamma=5, lags=12))
        for bps in (0, 5, 10, 15, 25, 50):
            na, nb = ea - bps * 1e-4 * ta, eb - bps * 1e-4 * tb
            summarize_pair(costs, universe, na + source_rf.to_numpy(), na, ta,
                           nb + source_rf.to_numpy(), nb, tb, cost_bps=bps)
        for low, high in ((2000, 2009), (2010, 2019)):
            mask = (pair.index.year >= low) & (pair.index.year <= high)
            summarize_pair(subperiods, universe, a[mask], ea[mask], ta[mask], b[mask], eb[mask], tb[mask],
                           subperiod=f"{low}-{high}")
        breaks.append({"universe": universe, "break_even_bps":
                       10000 * _crra_break_even_cost(a, ta, b, tb, gamma=5)})
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    frames = [pd.DataFrame(rows) for rows in (primary, bootstrap, hac, costs, subperiods, breaks)]
    names = ("primary_metrics_total", "paired_block_bootstrap_total", "hac_inference_total",
             "cost_sensitivity_total", "subperiod_metrics_total", "break_even_costs")
    for name, frame in zip(names, frames):
        frame.to_csv(out / (name + ".csv"), index=False)
    write_tables(out, frames, args.bootstrap_reps)
    if args.update_tables:
        if args.bootstrap_reps != 30000:
            raise ValueError("Published tables require exactly 30000 bootstrap replications")
        table_dir = ROOT / "results" / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for name in ("tab_pinnpi_mv.tex", "tab_empirical_robustness.tex"):
            (table_dir / name).write_bytes((out / name).read_bytes())
    manifest = {"gamma": 5, "bootstrap_reps": args.bootstrap_reps, "bootstrap_seed": 20260907,
                "block_length": 12, "hac_lags": 12, "strategy_sources": sources,
                "risk_free_source_sha256": RF_SHA256,
                "limits": "Conditional on fixed specifications; costs are ex-post; no solver-accuracy-matched superiority claim."}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Empirical tables and statistics: {out}")


if __name__ == "__main__":
    main()
