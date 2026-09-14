"""Export numerical comparison tables from hash-checked experiment records."""
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path

import report_highdim_local
import report_neural_duality

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'results/tables'


def upper(x, places=3):
    return format(Decimal.from_float(float(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING), f'.{places}f')


def save(name, header, rows, align):
    TABLES.mkdir(parents=True, exist_ok=True)
    content = ['% Generated from hash-checked result files; bounds rounded upward.',
               r'\begin{tabular}{'+align+r'}\toprule', header+r'\\\midrule']
    content += [' & '.join(row)+r'\\' for row in rows]
    content += [r'\bottomrule\end{tabular}', '']
    (TABLES/name).write_text('\n'.join(content), encoding='utf-8')


def build_tables():
    hd = json.loads((ROOT/'results/highdim_local/summary.json').read_text())
    dual = json.loads((ROOT/'results/neural_duality/summary.json').read_text())
    save('tab_highdim.tex', r'$d$ & PDE & Hybrid & Time-only hybrid & Cells & Verify (s)',
         [[str(x['state_dimension']), upper(x['two_sided_pde_ce_upper_bp']),
           upper(x['hybrid_ce_upper_bp']), upper(x['time_only_hybrid_ce_upper_bp']),
           str(x['accepted_local_cells']), f"{x['verification_seconds']:.2f}"] for x in hd['main_cases']], 'rrrrrr')
    save('tab_highdim_radius.tex', r'$d$ & PDE & Hybrid & $\ell_R$',
         [[str(x['state_dimension']), upper(x['two_sided_pde_ce_upper_bp']), upper(x['hybrid_ce_upper_bp']),
           upper(x['localization_upper'], 8)] for x in hd['smaller_cube_same_policy']], 'rrrr')
    ci = dual['mc_finest_sampling_95_interval_bp']
    save('tab_neural_duality.tex', 'Evaluation & CE gap (bp) & Status',
         [['Outward PDE and localization', upper(dual['rigorous_relative_ce_upper_bp']), 'Continuous certificate'],
          ['Duality Monte Carlo', f"{dual['mc_finest_gap_bp']:.3f}", f"95\\% sampling CI $[{ci[0]:.3f},{ci[1]:.3f}]$"],
          ['Duality PDE refinement', f"{dual['pde_finest_gap_bp']:.3f}", 'Numerical estimate']], 'lrl')
    print('Generated three numerical tables from the supplied records.')


def main():
    report_highdim_local.main()
    report_neural_duality.main()
    build_tables()


if __name__ == '__main__':
    main()
