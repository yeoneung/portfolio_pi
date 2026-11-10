"""Hash-checked tables for the correlated, noncyclic factor experiments."""
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import sys

from mpmath import iv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import highdim_network


def records():
    iv.dps = 60
    rows = []
    for d in (10, 20, 50):
        directory = ROOT/'results/highdim_network'
        path = directory/f'candidate_d{d}.json'
        candidate = json.loads(path.read_bytes())
        report = json.loads((directory/f'certificate_d{d}.json').read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() != report['candidate_sha256']:
            raise ValueError('Changed candidate')
        for source, digest in report['verifier_source_sha256'].items():
            if hashlib.sha256((ROOT/'experiments'/source).read_bytes()).hexdigest() != digest:
                raise ValueError('Changed certificate source: '+source)
        recomputed = highdim_network.combine(candidate, report['covers'])
        for key in ('relative_ce_loss_upper_bp', 'whole_space_optimal_factor_lower',
                    'whole_space_policy_factor_upper', 'localization_upper'):
            if recomputed[key] != report[key]:
                raise ValueError('Inconsistent certificate assembly: '+key)
        total_fit = (candidate['training']['seconds']+candidate['common_potential_fit_seconds']
                     +sum(b['fit']['seconds'] for b in candidate['blocks']))
        rows.append(dict(dimension=d, gamma=3, neighbor_correlation=.1,
                         relative_ce_loss_upper_bp=report['relative_ce_loss_upper_bp'],
                         local_cylinders_verified=report['local_cylinders_verified'],
                         accepted_cells=report['total_accepted_cells'],
                         final_candidate_construction_seconds=total_fit,
                         verification_seconds=report['cover_verification_seconds'],
                         localization_upper=report['localization_upper'],
                         candidate_sha256=report['candidate_sha256'],
                         certificate_sha256=hashlib.sha256((directory/f'certificate_d{d}.json').read_bytes()).hexdigest()))
    return rows


def table(rows):
    lines = [r'\begin{tabular}{rrrr}\toprule',
             r'$d$ & CE upper bound (bp) & Accepted cells & Verification (s)\\\midrule']
    for row in rows:
        bound = Decimal.from_float(row['relative_ce_loss_upper_bp']).quantize(Decimal('.001'), rounding=ROUND_CEILING)
        lines.append(f'{row["dimension"]} & {bound:.3f} & {row["accepted_cells"]:,} & {row["verification_seconds"]:.2f}'+r'\\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}', ''])


def main():
    rows = records()
    summary = dict(cases=rows, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results/highdim_network/summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    output = ROOT/'results/tables'
    output.mkdir(exist_ok=True, parents=True)
    (output/'tab_highdim_network.tex').write_text(table(rows), encoding='utf-8')
    print(table(rows))


if __name__ == '__main__':
    main()
