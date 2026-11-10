"""Resume unchanged-input time-slab verification with a different worker count.

Requires a checkpoint with recorded input hashes. Completed slabs are not
silently accepted from an arbitrary or mismatched progress file.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time

import verify_neural_separate_parallel as parallel
import verify_neural_separate as v


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 1<=args.workers<=16:p.error('Invalid worker count')
    candidate=json.loads(args.candidate.read_bytes());v.validate_candidate(candidate)
    c=json.loads(v.CHECKPOINT.read_bytes())
    checkpoint=json.loads(args.checkpoint.read_bytes())
    expected=checkpoint['verified_input_sha256']
    for name,digest in expected.items():
        path=(args.candidate if name==args.candidate.name else
              v.CHECKPOINT if name==v.CHECKPOINT.name else Path(__file__).with_name(name))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Changed verification input: '+name)
    if expected[v.CHECKPOINT.name]!=v.EXPECTED:
        raise ValueError('Changed fixed neural policy')
    reports=[None]*len(candidate['slabs'])
    for report in checkpoint['slabs']:
        i=report['slab_index']
        if not 0<=i<len(reports) or reports[i] is not None or not report['full_slab_covered']:
            raise ValueError('Duplicate or invalid saved slab')
        if any(report[k]!=candidate['slabs'][i][k] for k in ('left_exact','right_exact')):
            raise ValueError('Saved slab geometry mismatch')
        reports[i]=report
    missing=[i for i,r in enumerate(reports) if r is None]
    diag=args.candidate.with_name(args.candidate.stem+'_diagnostics.json')
    output=args.output or args.candidate.with_name(args.candidate.stem+'_certificate.json')
    protected=[args.candidate,args.checkpoint,diag,v.CHECKPOINT,Path(__file__)]
    if output.resolve() in [x.resolve() for x in protected]:
        raise ValueError('Output must not overwrite an input')
    padding,batch,depth=[checkpoint[k] for k in ('padding_proposal','batch_size','max_depth')]
    start=time.perf_counter();previous_seconds=checkpoint['elapsed_seconds']
    with ProcessPoolExecutor(max_workers=args.workers,initializer=parallel.initialize,
                             initargs=(str(v.CHECKPOINT),str(args.candidate),str(diag),padding,batch,depth)) as pool:
        futures=[pool.submit(parallel.verify_slab,i) for i in missing]
        for future in as_completed(futures):
            report=future.result();reports[report['slab_index']]=report
            done=sum(r is not None for r in reports)
            progress={'bounded_domain_certified':False,'whole_space_certified':False,
                      'completed_slabs':done,'total_slabs':len(reports),
                      'elapsed_seconds':previous_seconds+time.perf_counter()-start,
                      'slabs':[r for r in reports if r is not None]}
            output.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
            print(f'Covered {done}/{len(reports)} slabs; resumed {time.perf_counter()-start:.1f}s',flush=True)
    for name,digest in expected.items():
        path=(args.candidate if name==args.candidate.name else
              v.CHECKPOINT if name==v.CHECKPOINT.name else Path(__file__).with_name(name))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Input changed while resuming: '+name)
    result=parallel.aggregate(c,candidate,reports)
    hashes={**expected,Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result.update({'input_sha256':hashes,'candidate_sha256':hashes[args.candidate.name],
                   'checkpoint_sha256':v.EXPECTED,'padding_proposal':padding,
                   'workers':args.workers,'previous_workers':checkpoint['workers'],
                   'resume_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   'seconds':previous_seconds+time.perf_counter()-start})
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
