"""Run hash-preserved certificate cores with repository-local report outputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=('direct', 'welfare'))
    args, remaining = parser.parse_known_args(argv)
    if args.benchmark == 'direct':
        import direct_residual_certificate as core
    else:
        import portfolio_welfare as core
    # Preserve the verifier source bytes recorded in supplied certificates.
    # Only an optional presentation export is redirected at runtime.
    core.TABLE = ROOT / 'results' / 'tables' / core.TABLE.name
    core.TABLE.parent.mkdir(parents=True, exist_ok=True)
    return core.main(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
