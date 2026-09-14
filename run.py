"""Portable command entry point for portfolio experiments and verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "verify-neural-welfare": [["experiments/neural_log_barrier.py"],
                              ["experiments/neural_welfare_bound.py"]],
    "verify-neural-log": [["experiments/neural_log_barrier.py"]],
    "audit-neural-policy": [["experiments/neural_certificate_audit.py"]],
    "reproduce-neural-policy": [["experiments/constrained_benchmark.py", "--seeds", "11",
                                 "--output-dir", "results/neural_certificate",
                                 "--checkpoint-dir", "results/neural_certificate"],
                                ["experiments/neural_certificate_audit.py"]],
    "verify-local-policy": [["analysis/reproduce_local_policy.py"]],
    "fit-local-policy": [["analysis/reproduce_local_policy.py", "--refit"]],
    "verify-policy-loss": [["analysis/reproduce_policy_loss.py"]],
    "fit-policy-loss": [["analysis/reproduce_policy_loss.py", "--refit"]],
    "verify-welfare": [["analysis/run_certificate.py", "welfare", "--verify-only"]],
    "fit-welfare": [["analysis/run_certificate.py", "welfare"]],
    "localization-budget": [["analysis/localization_budget.py"]],
    "verify": [["analysis/run_certificate.py", "direct", "--verify-only"]],
    "fit-certificate": [["analysis/run_certificate.py", "direct"]],
    "report-tables": [["analysis/report_tables.py"]],
    "test-certificate": [["-m", "unittest", "discover", "-s", "tests", "-p", "test_direct_certificate.py", "-v"]],
    "test": [["-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"]],
    "empirical": [["analysis/reproduce_empirical.py"]],
    "ode": [["experiments/exact_pi_ode.py"]],
    "figures": [["figures/make_figures.py"], ["figures/make_newton_figure.py"]],
    "constrained": [["experiments/constrained_benchmark.py"]],
    "neural-diagnostics": [["experiments/pipinn_hard_certificate.py"]],
    "train-empirical": [["analysis/train_empirical.py"]],
}

def verify_manifest():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    failed = []
    for relative, expected in manifest["sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failed.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(relative)
    if failed:
        raise SystemExit("Missing or changed files: " + ", ".join(failed))
    print(f"Verified {len(manifest['sha256'])} supplied file hashes.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["manifest", *COMMANDS])
    args, extra = p.parse_known_args()
    if args.command == "manifest":
        if extra:
            p.error("manifest takes no additional arguments")
        verify_manifest()
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for command in COMMANDS[args.command]:
        subprocess.run([sys.executable, "-B", *command, *extra], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
