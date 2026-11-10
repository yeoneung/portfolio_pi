"""Report certified bounds separately from the existing numerical duality data."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/neural_separate_evaluation'


def main():
    old_path=ROOT/'results/neural_certificate/neural_welfare_bound.json'
    old=json.loads(old_path.read_bytes())
    coarse=json.loads((OUT/'whole_space_coarse.json').read_bytes())
    refined=json.loads((OUT/'whole_space_refined.json').read_bytes())
    for result in (old,coarse,refined):
        if not result['whole_space_certified']:
            raise ValueError('Completed whole-space certificates required')
    if coarse['checkpoint_sha256']!=refined['checkpoint_sha256'] or old['checkpoint_sha256']!=refined['checkpoint_sha256']:
        raise ValueError('The policies must be identical')
    pde=json.loads((ROOT/'results/neural_duality/pde_time_refinement.json').read_bytes())['levels'][-1]
    previous_pde=json.loads((ROOT/'results/neural_duality/pde_refinement.json').read_bytes())['levels'][-1]
    if pde['dz']!=previous_pde['dz'] or pde['dt']*2!=previous_pde['dt']:
        raise ValueError('The independent time refinements must share a spatial grid')
    richardson=2*pde['policy_factor']-previous_pde['policy_factor']
    separate_factor=json.loads((OUT/'adapted_policy_diagnostics.json').read_bytes())['policy_factor_sampled']
    coarse_factor=json.loads((OUT/'march_expanded_refined_diagnostics.json').read_bytes())['policy_factor_sampled']
    fitting={name:json.loads((OUT/(name+'.json')).read_bytes())['fitting_seconds'] for name in ('adapted_policy','adapted_bellman')}
    verification={name:json.loads((OUT/(name+'_certificate.json')).read_bytes())['seconds'] for name in ('adapted_policy','adapted_bellman')}
    workers={name:{k:json.loads((OUT/(name+'_certificate.json')).read_bytes()).get(k)
                   for k in ('workers','previous_workers')} for name in ('adapted_policy','adapted_bellman')}
    raw=refined['relative_ce_loss_upper_bp']
    rounded=str(Decimal.from_float(raw).quantize(Decimal('.001'),rounding=ROUND_CEILING))
    summary={
        'same_checkpoint':True,'policy_updates_after_checkpoint':0,
        'original_direct_neural_bound_bp':old['relative_ce_loss_upper_bp'],
        'separate_coarse_bound_bp':coarse['relative_ce_loss_upper_bp'],
        'separate_refined_bound_bp':raw,'refined_bound_bp_rounded_up':rounded,
        'improvement_factor_over_original_direct_certificate':old['relative_ce_loss_upper_bp']/raw,
        'duality_pde_estimate_bp_NOT_certificate':10000*pde['relative_ce_duality_gap'],
        'independent_pde_time_extrapolation_NOT_certificate':richardson,
        'separate_evaluation_factor_NOT_certificate':separate_factor,
        'difference_from_independent_pde_extrapolation':separate_factor-richardson,
        'fixed_policy_factor_change_on_refinement_NOT_certificate':abs(separate_factor-coarse_factor),
        'budget_bp':refined['budget_bp'],
        'fitting_seconds_by_candidate':fitting,'verification_wall_seconds_by_candidate':verification,
        'verification_workers_by_candidate':workers,
        'scope':'30 assets, one state variable, gamma=2, T=3, initial center',
        'whole_space_certified':True,
        'method':'frozen-policy upper comparison plus independent Bellman lower comparison and exit bounds',
        'manuscript_updated':False,'arxiv_manuscript_updated':False,
        'public_repository_updated':False,
        'inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (old_path,OUT/'whole_space_coarse.json',OUT/'whole_space_refined.json')},
        'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.1,4.2),layout='constrained')
    values=[old['relative_ce_loss_upper_bp'],coarse['relative_ce_loss_upper_bp'],raw]
    ax.bar(range(3),values,color=['#9aa4b2','#6d9dcc','#236ba0'],width=.57)
    ax.set_yscale('log');ax.set_ylim(1,2000)
    ax.set_xticks(range(3),['Direct neural\ncertificate','Separate evaluation\ncoarse','Separate evaluation\nrefined'])
    ax.set_ylabel('Certified relative CE loss upper bound (bp)')
    ax.set_title('Same frozen 30-asset neural policy')
    for i,val in enumerate(values):
        label=str(Decimal.from_float(val).quantize(Decimal('.001'),rounding=ROUND_CEILING))
        ax.text(i,val*1.1,label,ha='center',va='bottom')
    dual=10000*pde['relative_ce_duality_gap']
    ax.axhline(dual,color='#666666',linestyle='--',linewidth=1)
    ax.text(-.43,dual*.86,f'Duality PDE estimate: {dual:.2f} bp (not certified)',fontsize=9,color='#555555',va='top')
    ax.grid(axis='y',which='major',alpha=.17);ax.set_axisbelow(True)
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT/('certificate_comparison.'+suffix),dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
