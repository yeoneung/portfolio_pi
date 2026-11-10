"""Plot full-space time-slab residual bounds from completed certificates."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/neural_separate_evaluation'


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,3.2),layout='constrained')
    inputs={}
    for name,field,sign,label,color in (
            ('adapted_policy','normalized_residual_lower',-1,r'Fixed-policy envelope $\varepsilon_\pi$','#236ba0'),
            ('adapted_bellman','normalized_residual_upper',1,r'Bellman envelope $\varepsilon_*$','#bc542e')):
        path=OUT/(name+'_certificate.json')
        certificate=json.loads(path.read_bytes())
        if not certificate['bounded_domain_certified']:
            raise ValueError('Incomplete residual verification')
        candidate=OUT/(name+'.json')
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=certificate['candidate_sha256']:
            raise ValueError('Changed candidate')
        rows=certificate['slabs']
        edges=np.array([3*float(F(r['left_exact'])) for r in rows]+[3.])
        values=np.array([max(0.,sign*r[field]) for r in rows])
        ax.stairs(values,edges,label=label,color=color,linewidth=1.35)
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    # A symmetric-log time axis includes tau=0 and resolves the terminal layer.
    ax.set_xscale('symlog',linthresh=1e-5,linscale=.3)
    ax.set_yscale('log');ax.set_xlim(0,3)
    ax.set_xlabel(r'Remaining time $\tau$ (years)')
    ax.set_ylabel('Normalized residual envelope')
    ax.set_xticks([0,1e-4,1e-3,1e-2,.1,1,3],['0',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$','0.1','1','3'])
    ax.grid(which='major',alpha=.16);ax.legend(loc='upper right',frameon=False,fontsize=9)
    for suffix in ('pdf','png'):
        fig.savefig(OUT/('neural_residual_envelopes.'+suffix),dpi=180)
    plt.close(fig)
    (OUT/'residual_figure_sources.json').write_text(json.dumps({'input_sha256':inputs,
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
