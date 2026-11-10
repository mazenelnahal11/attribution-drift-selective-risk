"""Reconstruct source bootstrap resamples and selection-window statistics from raw arrays."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,wasserstein_distance

NAMES=('disagreement','js_only','rank_only','lime_noise','entropy','msp_uncertainty','energy','expected_cost','input_distance','single_A','single_B','signed_difference','input_KS','input_RFF_MMD')

def audit_unit(task):
    unit,cfg=task;unit=Path(unit)
    ref=np.load(unit/'reference_explanations.npz');cal=np.load(unit/'calibration_explanations.npz');boot=np.load(unit/'bootstrap_statistics.npz')
    third=cfg['calibration_n']//3
    RFF=lambda X:np.sqrt(2/64)*np.cos(X@boot['rff_w']+boot['rff_b'])
    reference_rff=RFF(ref['X']).mean(0)
    def statistics(indices):
        values=[wasserstein_distance(ref['scalar'][:,j],cal['scalar'][indices,j]) for j in range(9)]
        for feature in ('A','B',None):
            r=ref[feature] if feature else ref['A']-ref['B'];v=cal[feature][indices] if feature else (cal['A']-cal['B'])[indices]
            values.append(np.mean([wasserstein_distance(r[:,j],v[:,j]) for j in range(r.shape[1])]))
        values.append(max(ks_2samp(ref['X'][:,j],cal['X'][indices,j]).statistic for j in range(ref['X'].shape[1])))
        values.append(np.sum((RFF(cal['X'][indices]).mean(0)-reference_rff)**2))
        return np.asarray(values)
    checks=0;failures=[]
    for W in cfg['windows']:
        parts=('bootstrap',cfg['monitor_seed'],W)
        seed=int(hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        for replicate in range(cfg['bootstrap_n']):
            block=min(cfg['block_length'],third)
            starts=rng.integers(0,third-block+1,math.ceil(W/block))
            indices=np.concatenate([np.arange(s,s+block) for s in starts])[:W]+third
            expected=statistics(indices);actual=boot[f'W{W}'][replicate]
            checks+=len(NAMES)
            if not np.allclose(expected,actual,rtol=2e-9,atol=2e-10):
                failures.append(f'bootstrap/W{W}/replicate{replicate}')
    windows=pd.read_csv(unit/'calibration_windows.csv',float_precision='round_trip')
    for (W,end),group in windows.groupby(['window','end']):
        indices=np.arange(end-W,end)+2*third
        expected=statistics(indices)
        actual=group.set_index('method').loc[list(NAMES)].statistic.to_numpy()
        checks+=len(NAMES)
        if not np.allclose(expected,actual,rtol=2e-9,atol=2e-10):
            failures.append(f'selection_window/W{W}/end{end}')
    return {'unit':unit.as_posix(),'checks':checks,'failures':failures}

def audit(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text())
    tasks=[(str(u),cfg) for u in sorted((root/'units').glob('*/*/seed_*'))]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4,max(1,(os.cpu_count() or 2)//2))) as pool:
        for result in pool.map(audit_unit,tasks):
            result['unit']=Path(result['unit']).relative_to(root/'units').as_posix()
            results.append(result)
            print('Source bootstrap verified',result['unit'],flush=True)
    report={'status':'passed' if all(not r['failures'] for r in results) else 'failed','checks':sum(r['checks'] for r in results),'units':results,'scope':'Every source bootstrap replicate and all source selection-window statistics; all14 detectors, all3 windows.'}
    (root.parent/(root.name+'_calibration_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='units'},indent=2))
    return report

if __name__=='__main__':
    report=audit(sys.argv[1]);raise SystemExit(0 if report['status']=='passed' else 1)
