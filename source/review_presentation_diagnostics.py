"""Read-only presentation diagnostics for the frozen confirmation run; zero model fits.

Usage: python review_presentation_diagnostics.py RAW_RUN ANALYSIS_DIRECTORY OUTPUT_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

p=argparse.ArgumentParser()
p.add_argument('root');p.add_argument('analysis');p.add_argument('out')
args=p.parse_args()
RUN=Path(args.root);ANALYSIS=Path(args.analysis);WORK=Path(args.out)
WORK.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256((RUN/'engine.py').read_bytes()).hexdigest()=='53a0d22652fdff2683dc9053f8ad072f09958b79b518f0ba5edb63ccebcbf4e4'
read=lambda p:pd.read_csv(p,float_precision='round_trip')
tail=[]
for seed in (11,29,47,71,101):
    unit=RUN/'units/interaction/xgboost'/f'seed_{seed}'
    mapper=json.loads((unit/'quantile_mapping.json').read_text())
    Dmax=max(mapper['sorted_D']);umax=max(mapper['sorted_u'])
    ref=np.load(unit/'reference_explanations.npz')
    for stream in ('covariate_interaction','local_boundary_shift'):
        rows=read(unit/(stream+'_rows.csv')).iloc[1200:]
        D=rows.disagreement.to_numpy();matched=rows.matched_disagreement.to_numpy()
        exceed=D>Dmax;endpoint=matched==umax
        true_cost=np.where(rows.p<0.5,5*rows.true_eta,1-rows.true_eta)
        tail.append({'seed':seed,'stream':stream,'n':len(rows),'mapping_D_max':Dmax,'reference_D_max':float(ref['scalar'][:,0].max()),'target_D_max':D.max(),'mapping_u_max':umax,'D_exceeds_mapping_max_n':int(exceed.sum()),'D_exceeds_mapping_max_fraction':float(exceed.mean()),'at_upper_endpoint_n':int(endpoint.sum()),'at_upper_endpoint_fraction':float(endpoint.mean()),'distinct_D':int(rows.disagreement.nunique()),'distinct_matched_D':int(rows.matched_disagreement.nunique()),'distinct_CP':int(rows.risk_W100_cost_preserved.nunique()),'D_vs_true_cost_spearman':float(spearmanr(D,true_cost).statistic),'matched_D_vs_true_cost_spearman':float(spearmanr(matched,true_cost).statistic),'mean_gamma':float(rows.gamma_directional_W100.mean()),'active_fraction':float((rows.gamma_directional_W100>0).mean())})
pd.DataFrame(tail).to_csv(WORK/'interaction_tail_diagnostics.csv',index=False)
selection=read(RUN/'selection_all.csv').drop_duplicates(['dataset','model','seed','stream','scope','window','method'])
controls=selection.loc[(selection.scope=='fully_shifted') & (selection.stream.str.startswith('covariate')) & selection.dataset.isin(['sea','hyperplane','adult_numeric'])]
window_rows=[]
for W in (50,100,250):
    table=controls.loc[controls.window==W].pivot(index=['dataset','model','seed'],columns='method',values='aurc_cost')
    for method in ('naive_unscaled','cost_preserved'):
        delta=table[method]-table.expected_cost
        window_rows.append({'window':W,'method':method,'n':len(delta),'lower':int((delta<-1e-12).sum()),'higher':int((delta>1e-12).sum()),'ties':int((delta.abs()<=1e-12).sum()),'mean_delta':delta.mean()})
pd.DataFrame(window_rows).to_csv(WORK/'window_consistency.csv',index=False)
interaction=selection.loc[(selection.dataset=='interaction')&(selection.scope=='fully_shifted')&(selection.window==100)&selection.stream.isin(['covariate_interaction','local_boundary_shift'])]
ablations=interaction.groupby(['model','stream','method']).agg(aurc_mean=('aurc_cost','mean'),aurc_sd=('aurc_cost','std'),conditional_aurc_mean=('true_conditional_aurc','mean')).reset_index()
ablations.to_csv(WORK/'interaction_ablations.csv',index=False)
detect=read(RUN/'detection_all.csv')
detected=detect.loc[detect.stream.str.startswith('covariate')&detect.method.isin(['disagreement','ADWIN_confidence_native'])]
detected=detected.loc[((detected.method=='disagreement')&(detected.window==100))|((detected.method=='ADWIN_confidence_native')&(detected.window==0))]
alarms=detected.groupby('method').agg(streams=('seed','size'),detected=('detected','sum'),streams_with_pre_alarm=('any_pre_alarm','sum'),mean_pre_alarm_fraction=('pre_or_stationary_alarm_fraction','mean'),mean_delay_or_censor=('delay_or_censor_samples','mean')).reset_index()
alarms.to_csv(WORK/'adwin_comparison.csv',index=False)
ties=[]
primary=read(ANALYSIS/'table1_seed_results.csv')
for row in primary.loc[primary.dataset.isin(['sea','hyperplane','adult_numeric'])].itertuples():
    if abs(row.cost_preserved-row.expected_cost)>1e-12:
        continue
    rows=read(RUN/'units'/row.dataset/row.model/f'seed_{row.seed}'/(row.stream+'_rows.csv'))
    start=1800 if row.dataset=='hyperplane' else 1200
    post=rows.iloc[start:]
    ties.append({'dataset':row.dataset,'model':row.model,'seed':row.seed,'active_fraction':float((post.gamma_directional_W100>0).mean()),'scores_equal_c':bool(np.array_equal(post.risk_W100_cost_preserved,post.risk_W100_expected_cost))})
pd.DataFrame(ties).to_csv(WORK/'tie_activation.csv',index=False)
phase=read(ANALYSIS/'phase_diagnostics.csv')
phase=phase.loc[(phase.dataset=='interaction')&(phase.stream=='local_boundary_shift')]
mse=phase.groupby(['model','phase']).posterior_mse.mean()
result={'tail':tail,'windows':window_rows,'adwin':alarms.to_dict('records'),'ties':ties,'posterior_mse':{model:{'pre':mse[model,'pre'],'post':mse[model,'post'],'ratio':mse[model,'post']/mse[model,'pre']} for model in ('xgboost','mlp')},'frozen_engine_sha256':hashlib.sha256((RUN/'engine.py').read_bytes()).hexdigest(),'mode':'Read-only computations from frozen confirmation records; zero model fits.'}
(WORK/'review_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print(ablations.loc[ablations.method.isin(['expected_cost','naive_unscaled','cost_preserved','cost_only_two_sided','quantile_only_two_sided','cost_quantile_two_sided','naive_directional'])].to_string(index=False))
