"""Prepare or train a paper-universe portfolio solver using supplied inputs.

The selected specification and all model/solver settings come from the
provided resolved configuration. Input/output paths are resolved at runtime,
and every backend uses the common covariance-metric QP. An optional device
override is supported. PINN-PI is the default backend. Training is
separate from reproduction of the published fixed-path performance tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "code" / "empirical_src_bundle"
sys.path.insert(0, str(BUNDLE))

from dynalloc_v2.experiments import run_experiment
from dynalloc_v2.factor_zoo import build_candidate_panels, default_factor_zoo_v2_registry
from dynalloc_v2.schema import Config


def selected_candidate(project_name):
    candidates = sorted(default_factor_zoo_v2_registry(), key=lambda c: len(c.name), reverse=True)
    for candidate in candidates:
        if re.search(rf"(?:^|_){re.escape(candidate.name)}(?:_|$)", project_name):
            return candidate
    raise ValueError("Selected predictor specification is not identified by the source configuration")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", choices=("ff10", "ff12", "ff17", "ff25"), default="ff25")
    p.add_argument("--backend", choices=("pipinn", "pinn", "fdm", "fdm_pi"), default="pipinn")
    p.add_argument("--device", default=None)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    source = BUNDLE / "comparative" / f"{args.universe}_pls_tau3_fixed" / "selected_by_pipinn" / "rank_001" / f"run_as_{args.backend}" / "resolved_config.yaml"
    if not source.is_file():
        p.error(f"No supplied {args.backend} configuration for {args.universe}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["pipinn"]["policy_output_mode"] = "pure_qp"
    candidate = selected_candidate(payload["project"]["name"])
    base = BUNDLE / "base_bundles" / args.universe
    frames = {name: pd.read_csv(base / f"{name}_panel.csv", parse_dates=["date"]).set_index("date").sort_index()
              for name in ("returns", "macro", "ff3", "ff5", "bond")}
    common = frames["returns"].index
    for frame in frames.values():
        common = common.intersection(frame.index)
    frames = {name: frame.loc[common] for name, frame in frames.items()}
    split = payload["split"]
    train_dates = common[(common >= pd.Timestamp(split["train_start"])) & (common < pd.Timestamp(split["test_start"]))]
    panels = build_candidate_panels(candidate, **frames, train_dates=train_dates)
    output = ROOT / "results" / "empirical_training" / args.universe / args.backend
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, frame in (("returns", frames["returns"]), ("states", panels["states"]), ("factors", panels["factors"])):
        path = inputs / f"{name}_panel.csv"
        frame.reset_index(names="date").to_csv(path, index=False)
        payload["data"][f"{name}_csv"] = str(path)
    payload["project"]["name"] = f"{args.universe}_{args.backend}_{candidate.name}"
    payload["project"]["output_dir"] = str(output)
    if args.device:
        payload["pipinn"]["device"] = args.device
        payload["ppgdpo"]["device"] = args.device
    config = Config.model_validate(payload)
    target = output / "run_config.yaml"
    target.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    print(f"Prepared configuration: {target}")
    if not args.prepare_only:
        result = run_experiment(config)
        print(f"Training results: {result.output_dir}")


if __name__ == "__main__":
    main()
