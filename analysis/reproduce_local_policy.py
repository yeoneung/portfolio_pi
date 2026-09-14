"""Verify local fixed-policy evaluations and separate their error reductions."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
import local_policy_evaluation as local
import verify_local_policy as verify
import policy_loss_certificate as p
import portfolio_welfare as w
import finite_horizon_localization as finite

CASES = ((16, 8, 5, 4, "0.0000035"), (24, 8, 6, 4, "0.0000009"))
TABLE = ROOT/"results/tables/tab_local_policy.tex"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    local.OUT.mkdir(exist_ok=True)
    summary_path = local.OUT/"comparison_results.json"
    summary_path.write_text('{"certified": false, "reason": "comparison incomplete"}\n')
    TABLE.write_text("% Local-policy verification incomplete.\n\\textit{No certificate returned.}\n")
    policy_hash = hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest()
    old_path = p.OUT/"evaluation_32_64_6_certificate.json"
    old = json.loads(old_path.read_text())
    if not old["certified"] or old["policy_sha256"] != policy_hash or old["source_sha256"] != p.source_hashes():
        raise ValueError("Missing or stale global evaluation certificate; run verify-policy-loss")
    old_candidate = json.loads((p.OUT/"evaluation_32_64_6.json").read_text())
    if old["evaluation_sha256"] != hashlib.sha256((p.OUT/"evaluation_32_64_6.json").read_bytes()).hexdigest():
        raise ValueError("Global candidate hash mismatch")
    market = w.Market.from_json(old_candidate["market"])
    point = F(old["bounds"]["evaluation_initial_exact"])
    rows = [{"label": "Global $(32,64)$", "degrees_of_freedom": 33*65,
             "quadratic_localization_bounds": old["bounds"],
             "finite_horizon_bounds": finite.bounds(market, point, old["certificate"]),
             "verification_seconds": old["verification_seconds"], "certificate": old["certificate"]}]
    for ds, degree, level, rw, target in CASES:
        stem = f"evaluation_{ds}_{degree}_{level}_{rw}_p2_b3"
        if args.refit:
            subprocess.run([sys.executable, "-B", "experiments/local_policy_evaluation.py",
                            "--degree-time", str(ds), "--degree-space", str(degree),
                            "--level", str(level), "--reweights", str(rw),
                            "--time-power", "2", "--boundary-levels", "3"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-B", "experiments/verify_local_policy.py",
                        "--candidate", stem+".json", "--target-residual", target], cwd=ROOT, check=True)
        result = json.loads((local.OUT/(stem+"_certificate.json")).read_text())
        if not result["certified"] or result["policy_sha256"] != policy_hash or result["source_sha256"] != verify.source_hashes():
            raise ArithmeticError("Incomplete or stale local evaluation certificate")
        if result["evaluation_sha256"] != hashlib.sha256((local.OUT/(stem+".json")).read_bytes()).hexdigest():
            raise ArithmeticError("Local candidate changed")
        diagnostic = json.loads((local.OUT/(stem+"_diagnostics.json")).read_text())
        point = F(result["bounds"]["evaluation_initial_exact"])
        rows.append({"label": f"Local $({ds},{result['space_intervals']})$",
                     "degrees_of_freedom": (ds+1)*(degree-1)*result["space_intervals"],
                     "fitting_seconds_recorded": diagnostic["fitting_seconds"],
                     "quadratic_localization_bounds": finite.bounds(market, point, result["certificate"], localizer=w.localization),
                     "finite_horizon_bounds": result["bounds"], **result})
    if hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest() != policy_hash:
        raise ArithmeticError("The deployed policy changed")
    direct = json.loads((ROOT/"results/portfolio_welfare/welfare_results.json").read_text())
    if not direct["certified"] or direct["results"][0]["candidate_sha256"] != policy_hash:
        raise ArithmeticError("Direct baseline does not match the policy")
    fine = rows[-1]["finite_horizon_bounds"]["relative_ce_loss_bp_upper"]
    reduction_global = w.lower(w.mpiv(rows[0]["quadratic_localization_bounds"]["relative_ce_loss_bp_upper"])/w.mpiv(fine))
    reduction_direct = w.lower(w.mpiv(direct["results"][0]["bounds"]["relative_ce_loss_bp_upper"])/w.mpiv(fine))
    summary = {"certified": True, "policy_sha256": policy_hash, "policy_updates": 0,
               "comparison": rows, "reduction_from_global_lower": reduction_global,
               "reduction_from_direct_lower": reduction_direct,
               "timing_scope": "recorded runs with different spaces and error targets; no matched-budget speed claim"}
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["% Generated by analysis/reproduce_local_policy.py; bounds rounded upward.",
             r"\begin{tabular}{lrrrr}\toprule",
             r"Evaluation & Coefficients & CE: quadratic & CE: finite-horizon & Verify (s)\\\midrule"]
    for row in rows:
        lines.append(row["label"]+f" & {row['degrees_of_freedom']} & $"+
                     w.exact.upper_sci(row["quadratic_localization_bounds"]["relative_ce_loss_bp_upper"])+"$ & $"+
                     w.exact.upper_sci(row["finite_horizon_bounds"]["relative_ce_loss_bp_upper"])+f"$ & {row['verification_seconds']:.1f}"+r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    TABLE.write_text("\n".join(lines)+"\n")
    print(json.dumps({"certified": True, "reduction_from_global_lower": reduction_global,
                      "reduction_from_direct_lower": reduction_direct,
                      "uniform_inner_bounds": rows[-1]["uniform_inner_bounds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
