"""Independent reconstruction of Path A+ records; the experiment engine is never imported.

Usage: python audit_v2.py RUN_DIRECTORY [--full]
Requires numpy, scipy, pandas, sklearn; full also verifies saved torch/XGB models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit,rel_entr
from scipy.stats import rankdata,wasserstein_distance,ks_2samp
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

SCALARS=('disagreement','js_only','rank_only','lime_noise','entropy','msp_uncertainty','energy','expected_cost','input_distance')
DETECTORS=SCALARS+('single_A','single_B','signed_difference','input_KS','input_RFF_MMD')
POLICIES=('expected_cost','naive_unscaled','cost_preserved','cost_quantile_two_sided','cost_only_two_sided','quantile_only_two_sided','naive_directional','static_disagreement','entropy','msp_uncertainty','energy','input_distance','within_surrogate_disagreement')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reconstruct_disagreement(A,B):
    aa,bb=np.abs(A),np.abs(B)
    def mass(v):
        s=v.sum(1,keepdims=True)
        return np.divide(v,s,out=np.full_like(v,1/v.shape[1]),where=s>1e-12)
    p,q=mass(aa),mass(bb);m=(p+q)/2
    js=np.clip((rel_entr(p,m).sum(1)+rel_entr(q,m).sum(1))/(2*np.log(2)),0,1)
    ra,rb=rankdata(-aa,axis=1),rankdata(-bb,axis=1)
    top=np.zeros(A.shape,dtype=bool)
    for v in (aa,bb):
        indices=np.argsort(-v,axis=1,kind='stable')[:,:min(3,A.shape[1])]
        np.put_along_axis(top,indices,True,axis=1)
    numerator=np.zeros(len(A));denominator=numerator.copy()
    for i in range(A.shape[1]):
        for j in range(i+1,A.shape[1]):
            sa,sb=np.sign(ra[:,i]-ra[:,j]),np.sign(rb[:,i]-rb[:,j])
            gap=np.where(sa==sb,0,np.where((sa==0)|(sb==0),.5,1))
            weight=(top[:,i]&top[:,j])/(np.minimum(ra[:,i],rb[:,i])*np.minimum(ra[:,j],rb[:,j]))
            numerator+=weight*gap;denominator+=weight
    rank=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    return .5*(js+rank),js,rank

def aurc(loss,score):
    order=np.argsort(score,kind='stable');s=np.asarray(score)[order];l=np.asarray(loss)[order]
    ends=np.r_[np.flatnonzero(np.diff(s))+1,len(s)]
    coverage=ends/len(s);risk=np.cumsum(l)[ends-1]/ends
    return np.dot(np.diff(np.r_[0.,coverage]),risk)

def independent_scores(rows,two,directional,sorted_D,sorted_u):
    p=rows.p.to_numpy();D=rows.disagreement.to_numpy()
    u=np.minimum(p,1-p);w=np.where(p<.5,1.,.2);c=w*u
    q=np.searchsorted(sorted_D,D,side='right')/len(sorted_D)
    mapped=sorted_u[np.maximum(0,np.minimum(len(sorted_u)-1,np.ceil(q*len(sorted_u)).astype(int)-1))]
    return {'expected_cost':c,'naive_unscaled':(1-two)*c+two*D,'cost_preserved':w*((1-directional)*u+directional*mapped),
            'cost_quantile_two_sided':w*((1-two)*u+two*mapped),'cost_only_two_sided':w*((1-two)*u+two*D),
            'quantile_only_two_sided':(1-two)*c+two*mapped,'naive_directional':(1-directional)*c+directional*D,
            'static_disagreement':D,'entropy':rows.entropy.to_numpy(),'msp_uncertainty':1-np.maximum(p,1-p),'energy':rows.energy.to_numpy(),
            'input_distance':rows.input_distance.to_numpy(),'within_surrogate_disagreement':rows.lime_noise.to_numpy()},mapped

def gammas(windows,n,reference_mean):
    two=np.zeros(n);directional=two.copy()
    records=list(windows.loc[windows.method=='disagreement'].sort_values('end').itertuples())
    for k,row in enumerate(records):
        start=row.end-1;stop=records[k+1].end-1 if k+1<len(records) else n
        value=max(0.,min(1.,1-row.threshold/max(row.statistic,1e-12))) if row.statistic>row.threshold else 0.
        two[start:stop]=value
        if row.mean_disagreement>reference_mean:
            directional[start:stop]=value
    return two,directional

def audit(root,full=False):
    root=Path(root);cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'run_status.json').read_text())
    failures=[];count=0;unit_summary=[];fingerprints={};case_fingerprints={}
    def check(name,ok):
        nonlocal count
        count+=1
        if not bool(ok):
            failures.append(name)
    def near(a,b):
        return np.allclose(a,b,rtol=2e-8,atol=2e-9,equal_nan=True)
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    check('run complete',status['status']=='complete')
    check('engine identity',sha(root/'engine.py')==status['engine_sha256'])
    manifest_bytes=(root/'revision_data_manifest.json').read_bytes()
    # The frozen Windows source bytes use CRLF; run exports serialize JSON using host newlines.
    manifest_lf=manifest_bytes.replace(b'\r\n',b'\n')
    valid_manifest_hashes={hashlib.sha256(blob).hexdigest() for blob in (manifest_bytes,manifest_lf,manifest_lf.replace(b'\n',b'\r\n'))}
    check('data manifest identity (host newline normalization)',status['data_manifest_sha256'] in valid_manifest_hashes)
    # JSON serialization uses stable pretty printing for the archived manifest.
    manifest=json.loads((root/'file_manifest.json').read_text())
    for name,entry in manifest.items():
        check('file checksum:'+name,sha(root/name)==entry['sha256'])
    excluded=set(json.loads((root/'revision_data_manifest.json').read_text())['adult_excluded_deployment_ids'])
    for unit in sorted((root/'units').glob('*/*/seed_*')):
        key=unit.relative_to(root/'units').as_posix();dataset,model,_=key.split('/')
        before=count;source=np.load(unit/'source_split.npz');ref=np.load(unit/'reference_explanations.npz')
        cal=read(unit/'calibration_rows.csv');gate=read(unit/'gate_rows.csv')
        source_ids=set()
        for role in ('train','gate','reference','calibration'):
            ids=set(source[role+'_ids']);check(key+':source role '+role,len(ids)==len(source[role+'_ids']) and ids.isdisjoint(source_ids));source_ids.update(ids)
        third=cfg['calibration_n']//3
        check(key+':three calibration roles',cal.calibration_role.tolist()==['mapping']*third+['detector']*third+['selection']*third)
        mapping=json.loads((unit/'quantile_mapping.json').read_text());sorted_D=np.sort(cal.disagreement.to_numpy()[:third]);sorted_u=np.sort(np.minimum(cal.p,1-cal.p).to_numpy()[:third])
        check(key+':mapping fit identifiers',mapping['fit_ids']==cal.id.iloc[:third].tolist() and mapping['uses_labels'] is False)
        check(key+':mapping fit values',near(mapping['sorted_D'],sorted_D) and near(mapping['sorted_u'],sorted_u))
        boot=np.load(unit/'bootstrap_statistics.npz');thresholds=read(unit/'detector_thresholds.csv');cutoffs=read(unit/'selection_thresholds.csv');cw=read(unit/'calibration_windows.csv')
        reference_mean=ref['scalar'][:,0].mean()
        for W in cfg['windows']:
            saved=thresholds.loc[thresholds.window==W].set_index('method').loc[list(DETECTORS)].threshold.to_numpy()
            check(key+f':W{W} bootstrap cutoff',near(np.quantile(boot[f'W{W}'],.95,axis=0,method='higher'),saved))
            two,directional=gammas(cw.loc[cw.window==W],third,reference_mean)
            score_map,_=independent_scores(cal.iloc[2*third:],two,directional,sorted_D,sorted_u)
            for row in cutoffs.loc[cutoffs.window==W].itertuples():
                check(key+f':source cutoff {W}/{row.method}/{row.review_budget}',near(np.quantile(score_map[row.method],1-row.review_budget,method='higher'),row.threshold))
        quality=json.loads((unit/'quality.json').read_text())
        check(key+':source accuracy',near(balanced_accuracy_score(gate.y,gate.p>=.5),quality['balanced_accuracy']))
        check(key+':source gates',quality['all_passed'] and all(quality['gates'].values()))
        streams=json.loads((unit/'stream_manifest.json').read_text());windows=read(unit/'monitor_windows.csv');metrics=read(unit/'selection_metrics.csv');detection=read(unit/'detection_metrics.csv')
        cache={};fit_probs=[]
        for stream,meta in streams.items():
            rows=read(unit/f'{stream}_rows.csv');arrays=np.load(unit/f'{stream}_explanations.npz');cache[stream]=(rows,arrays)
            check(key+':'+stream+' disjoint identifiers',set(rows.id).isdisjoint(source_ids) and rows.id.nunique()==len(rows))
            if dataset=='adult_numeric':
                check(key+':fresh Adult '+stream,set(rows.id).isdisjoint(excluded))
            p=expit(arrays['margin']);check(key+':'+stream+' probabilities',near(rows.p,p))
            D,js,rank=reconstruct_disagreement(arrays['A'],arrays['B'])
            noise,_,_=reconstruct_disagreement(arrays['B'],arrays['B2'])
            check(key+':'+stream+' raw disagreement',near(rows.disagreement,D) and near(rows.js_only,js) and near(rows.rank_only,rank) and near(rows.lime_noise,noise))
            check(key+':'+stream+' scalar consistency',near(rows[list(SCALARS)].to_numpy(),arrays['scalar']))
            local_windows=windows.loc[windows.stream==stream]
            for W in cfg['windows']:
                win=local_windows.loc[local_windows.window==W]
                two,directional=gammas(win,len(rows),reference_mean)
                check(key+f':{stream}/W{W} causal actions',near(two,rows[f'gamma_two_sided_W{W}']) and near(directional,rows[f'gamma_directional_W{W}']))
                score_map,mapped=independent_scores(rows,two,directional,sorted_D,sorted_u)
                check(key+f':{stream}/W{W} quantile transform',near(mapped,rows.matched_disagreement))
                for method,scores in score_map.items():
                    check(key+f':{stream}/W{W} score:{method}',near(scores,rows[f'risk_W{W}_{method}']))
                # All scalar-D windows; every detector and dimension at the primary window.
                tested=win if W==cfg['primary_window'] else win.loc[win.method=='disagreement']
                for end,group in tested.groupby('end'):
                    start=int(group.start.iloc[0]);current=arrays['scalar'][start:end]
                    values={name:wasserstein_distance(ref['scalar'][:,j],current[:,j]) for j,name in enumerate(SCALARS)}
                    if W==cfg['primary_window']:
                        for name,feature in (('single_A','A'),('single_B','B'),('signed_difference',None)):
                            r=ref[feature] if feature else ref['A']-ref['B'];s=arrays[feature][start:end] if feature else (arrays['A']-arrays['B'])[start:end]
                            values[name]=np.mean([wasserstein_distance(r[:,j],s[:,j]) for j in range(r.shape[1])])
                        values['input_KS']=max(ks_2samp(ref['X'][:,j],arrays['X'][start:end,j]).statistic for j in range(ref['X'].shape[1]))
                        feature=lambda X:np.sqrt(2/64)*np.cos(X@boot['rff_w']+boot['rff_b'])
                        values['input_RFF_MMD']=np.sum((feature(ref['X']).mean(0)-feature(arrays['X'][start:end]).mean(0))**2)
                    check(key+f':{stream}/W{W}/end{end} window statistics',all(near(values[r.method],r.statistic) and r.alarm==(r.statistic>r.threshold) for r in group.itertuples()))
                    check(key+f':{stream}/W{W}/end{end} mean direction',near(current[:,0].mean(),group.mean_disagreement.iloc[0]) and near(reference_mean,group.reference_mean_disagreement.iloc[0]))
                for row in metrics.loc[(metrics.stream==stream)&(metrics.window==W)].itertuples():
                    take=np.arange(len(rows))>=meta['transition_end'] if row.scope=='fully_shifted' else np.ones(len(rows),dtype=bool)
                    y=rows.y.to_numpy()[take];pr=rows.p.to_numpy()[take];score=score_map[row.method][take]
                    losses=np.where((y==1)&(pr<.5),5.,np.where((y==0)&(pr>=.5),1.,0.));accept=score<=row.threshold;naccept=accept.sum()
                    check(key+f':{stream}/{W}/{row.scope}/{row.method}/{row.review_budget_source} metrics',near(aurc(losses,score),row.aurc_cost) and naccept==row.accepted and near(accept.mean(),row.coverage) and near(losses[accept].mean() if naccept else np.nan,row.selective_cost_risk))
                    if 'true_eta' in rows:
                        eta=rows.true_eta.to_numpy()[take];conditional=np.where(pr<.5,5*eta,1-eta)
                        check(key+':conditional risk '+str(row.Index),near(aurc(conditional,score),row.true_conditional_aurc))
                for method,group in win.groupby('method'):
                    defined=group.loc[group.phase.isin(['pre','post'])]
                    auc=roc_auc_score(defined.phase=='post',defined.statistic) if defined.phase.nunique()==2 else np.nan
                    observed=detection.loc[(detection.stream==stream)&(detection.window==W)&(detection.method==method)].iloc[0]
                    check(key+f':{stream}/{W}/{method} detection AUROC',near(auc,observed.drift_auroc))
            fit_probs.append(rows.p.to_numpy().tobytes())
        if 'label_only' in cache:
            r,a=cache['stationary'];s,b=cache['label_only']
            check(key+':paired label-only observability',np.array_equal(a['X'],b['X']) and np.array_equal(a['A'],b['A']) and np.array_equal(a['B'],b['B']) and near(r.filter(regex='^risk_|^gamma_|^p$').to_numpy(),s.filter(regex='^risk_|^gamma_|^p$').to_numpy()))
        if dataset=='interaction':
            check(key+':paired boundary inputs',np.array_equal(cache['covariate_interaction'][1]['X'],cache['local_boundary_shift'][1]['X']))
        checkpoint=unit/('model.ubj' if model=='xgboost' else 'model.pt')
        fingerprints[key]={'checkpoint_sha256':sha(checkpoint),'probability_sha256':hashlib.sha256(b''.join(fit_probs)).hexdigest()}
        case_fingerprints[key]=hashlib.sha256(b''.join(cache[s][1]['X'].tobytes()+cache[s][0].id.to_numpy().astype('U').tobytes() for s in sorted(cache))).hexdigest()
        if full:
            X=next(iter(cache.values()))[1]['X'];stored=next(iter(cache.values()))[1]['margin']
            if model=='xgboost':
                from xgboost import XGBClassifier
                m=XGBClassifier();m.load_model(checkpoint);replayed=m.predict(X,output_margin=True)
                # UBJ stores the fitted trees; reproduce the fit to verify training settings.
                from sklearn.preprocessing import StandardScaler
                Z=StandardScaler().fit_transform(source['train_X_raw']).astype('float32')
                refit=XGBClassifier(n_estimators=cfg['xgb_trees'],max_depth=cfg['xgb_depth'],learning_rate=.05,
                                    subsample=.8,colsample_bytree=.8,reg_lambda=1.,objective='binary:logistic',
                                    eval_metric='logloss',tree_method='hist',n_jobs=cfg['threads'],random_state=int(key.split('_')[-1]))
                refit.fit(Z,source['train_y'])
                check(key+':stochastic fit settings reproduce fitted model',near(refit.predict(X,output_margin=True),stored))
            else:
                import torch
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                m=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,32),torch.nn.SiLU(),torch.nn.Linear(32,1));m.load_state_dict(saved['state_dict']);m.eval()
                with torch.no_grad():
                    replayed=m(torch.tensor(X,dtype=torch.float32)).flatten().numpy()
            check(key+':checkpoint reproduces saved margins',near(replayed,stored))
        unit_summary.append({'unit':key,'checks':count-before,'source_balanced_accuracy':quality['balanced_accuracy']})
        print('Audited',key,flush=True)
    diversity=[]
    for dataset in cfg['datasets']:
        for model in cfg['models']:
            keys=[k for k in fingerprints if k.startswith(dataset+'/'+model+'/')]
            checkpoints=len({fingerprints[k]['checkpoint_sha256'] for k in keys});probabilities=len({fingerprints[k]['probability_sha256'] for k in keys});cases=len({case_fingerprints[k] for k in keys})
            check(dataset+'/'+model+' fit-seed diversity',checkpoints==probabilities==len(cfg['seeds']) and cases==1)
            diversity.append({'dataset':dataset,'model':model,'distinct_checkpoints':checkpoints,'distinct_probabilities':probabilities,'distinct_case_sets':cases})
    result={'status':'passed' if not failures else 'failed','checks':count,'failed_checks':failures,'units':unit_summary,'fit_diversity':diversity,'engine_sha256':sha(root/'engine.py'),'run_identity':status['identity'],'full_checkpoint_replay':full}
    (root.parent/(root.name+'_independent_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('units','fit_diversity')},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--full',action='store_true');args=parser.parse_args()
    result=audit(args.root,args.full)
    raise SystemExit(0 if result['status']=='passed' else 1)
