"""Generate manuscript tables and the requested two-panel figure from audited outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'sea':'SEA','hyperplane':'Hyperplane','adult_numeric':'Adult','electricity':'Electricity','interaction':'Interaction'}
MODELS={'xgboost':'XGB','mlp':'MLP'}
MAIN=('expected_cost','naive_unscaled','cost_preserved')
DETECT=('disagreement','signed_difference','single_A','single_B','entropy','input_KS','input_RFF_MMD')
COLORS={'expected_cost':'#222222','naive_unscaled':'#c2473b','cost_preserved':'#147d91'}
LABELS={'expected_cost':r'$c$','naive_unscaled':'Naive','cost_preserved':'CP'}
read=lambda p:pd.read_csv(p,float_precision='round_trip')

def md_table(columns,values):
    return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']+['|'+'|'.join(map(str,row))+'|' for row in values])

def curve(loss,score):
    order=np.argsort(score,kind='stable');s=score[order];l=loss[order]
    ends=np.r_[np.flatnonzero(np.diff(s))+1,len(s)]
    return ends/len(s),np.cumsum(l)[ends-1]/ends

def analyze(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((root.parent/(root.name+'_independent_audit.json')).read_text())
    assert audit['status']=='passed' and audit['full_checkpoint_replay']
    calibration_audit=json.loads((root.parent/(root.name+'_calibration_audit.json')).read_text())
    assert calibration_audit['status']=='passed'
    paired=read(root/'paired_aurc_contrasts.csv');selection=read(root/'selection_all.csv');detection=read(root/'detection_all.csv')
    primary=paired.loc[((paired.scope=='fully_shifted') & ((paired.dataset!='interaction') & paired.stream.str.startswith('covariate') | (paired.dataset=='interaction') & (paired.stream=='local_boundary_shift'))) | ((paired.dataset=='electricity') & (paired.scope=='all'))].copy()
    assert len(primary)==50 and primary.groupby(['dataset','model']).size().eq(5).all()
    primary.to_csv(out/'table1_seed_results.csv',index=False)
    table1=[]
    for (d,m),group in primary.groupby(['dataset','model'],sort=False):
        row={'dataset':d,'model':m,'scope':group.scope.iloc[0],'stream':group.stream.iloc[0],'seeds':len(group)}
        for method in MAIN:
            row[method+'_mean']=group[method].mean();row[method+'_sd']=group[method].std(ddof=1)
        for method in MAIN[1:]:
            delta=group[method]-group.expected_cost
            row[method+'_delta_mean']=delta.mean();row[method+'_delta_sd']=delta.std(ddof=1)
            row[method+'_lower']=int((delta<-1e-12).sum());row[method+'_higher']=int((delta>1e-12).sum());row[method+'_ties']=int((np.abs(delta)<=1e-12).sum())
        table1.append(row)
    table1=pd.DataFrame(table1);table1.to_csv(out/'table1_mean_seed_sd.csv',index=False)
    det=detection.loc[(detection.window==100)&detection.stream.str.startswith('covariate')]
    table2=det.loc[det.method.isin(DETECT)].groupby(['dataset','model','method']).agg(mean=('drift_auroc','mean'),sd=('drift_auroc','std'),seeds=('drift_auroc','count')).reset_index()
    table2.to_csv(out/'table2_mean_seed_sd.csv',index=False)
    fixed=selection.loc[(selection.window==100)&selection.method.isin(MAIN)].merge(primary[['dataset','model','seed','stream','scope']],on=['dataset','model','seed','stream','scope'],how='inner')
    fixed.to_csv(out/'fixed_threshold_seed_results.csv',index=False)
    fixed_summary=fixed.groupby(['dataset','model','method','review_budget_source']).agg(coverage_mean=('coverage','mean'),coverage_sd=('coverage','std'),accepted_cost_mean=('selective_cost_risk','mean'),accepted_cost_sd=('selective_cost_risk','std')).reset_index()
    fixed_summary.to_csv(out/'fixed_threshold_summary.csv',index=False)
    # Interaction covariate control and all ablations/windows remain available separately.
    cov_control=paired.loc[(paired.dataset=='interaction')&(paired.stream=='covariate_interaction')&(paired.scope=='fully_shifted')]
    cov_control.to_csv(out/'interaction_covariate_control.csv',index=False)
    all_summary=selection.drop_duplicates(['dataset','model','seed','stream','scope','window','method']).groupby(['dataset','model','stream','scope','window','method']).agg(aurc_mean=('aurc_cost','mean'),aurc_sd=('aurc_cost','std'),true_conditional_aurc_mean=('true_conditional_aurc','mean')).reset_index()
    all_summary.to_csv(out/'all_arms_windows_ablations.csv',index=False)
    fpr=detection.loc[(detection.window==100)&(detection.stream=='stationary')].groupby(['dataset','model','method']).agg(alarm_fraction_mean=('pre_or_stationary_alarm_fraction','mean'),alarm_fraction_sd=('pre_or_stationary_alarm_fraction','std'),streams_with_any_alarm=('any_pre_alarm','sum')).reset_index()
    fpr.to_csv(out/'stationary_alarm_summary.csv',index=False)
    controls=[];diagnostics=[]
    for unit in sorted((root/'units').glob('*/*/seed_*')):
        d,m,seed=unit.relative_to(root/'units').parts;seed=int(seed.split('_')[1])
        manifest=json.loads((unit/'stream_manifest.json').read_text())
        for stream,meta in manifest.items():
            rows=read(unit/f'{stream}_rows.csv');y=rows.y.to_numpy();p=rows.p.to_numpy()
            for phase,take in [('pre',np.arange(len(rows))<meta['onset']),('post',np.arange(len(rows))>=meta['transition_end'])] if meta['onset'] is not None else [('all',np.ones(len(rows),dtype=bool))]:
                r=rows.loc[take];pred=r.p.to_numpy()>=.5
                record={'dataset':d,'model':m,'seed':seed,'stream':stream,'phase':phase,'n':len(r),'D_mean':r.disagreement.mean(),'D_median':r.disagreement.median(),'gamma_two_sided_mean':r.gamma_two_sided_W100.mean(),'gamma_directional_mean':r.gamma_directional_W100.mean(),'directional_active_fraction':(r.gamma_directional_W100>0).mean(),'brier':np.mean((r.p-r.y)**2),'accuracy':np.mean(pred==r.y),'confidence':np.maximum(r.p,1-r.p).mean()}
                if 'true_eta' in r:
                    record['posterior_mse']=np.mean((r.p-r.true_eta)**2)
                    record['true_oracle_aurc']=np.nan
                    actual=np.where(r.p<.5,5*r.true_eta,1-r.true_eta)
                    from audit_v2 import aurc
                    record['true_oracle_aurc']=aurc(actual,actual)
                diagnostics.append(record)
        if 'label_only' in manifest:
            stationary=read(unit/'stationary_rows.csv');changed=read(unit/'label_only_rows.csv');take=np.arange(len(changed))>=manifest['label_only']['transition_end']
            controls.append({'dataset':d,'model':m,'seed':seed,'stationary_accuracy':np.mean((stationary.p.to_numpy()[take]>=.5)==stationary.y.to_numpy()[take]),'changed_label_accuracy':np.mean((changed.p.to_numpy()[take]>=.5)==changed.y.to_numpy()[take]),'identical_probabilities':np.array_equal(stationary.p,changed.p),'confidence':np.maximum(stationary.p,1-stationary.p).to_numpy()[take].mean()})
    pd.DataFrame(diagnostics).to_csv(out/'phase_diagnostics.csv',index=False);pd.DataFrame(controls).to_csv(out/'paired_label_controls.csv',index=False)
    pd.DataFrame(audit['fit_diversity']).to_csv(out/'fit_seed_diversity.csv',index=False)
    risk_rows=[]
    order=['sea','hyperplane','adult_numeric','interaction','electricity']
    for d in order:
        for m in ('xgboost','mlp'):
            r=table1.loc[(table1.dataset==d)&(table1.model==m)].iloc[0]
            risk_rows.append([NAMES[d],MODELS[m]]+[f'{r[k+"_mean"]:.4f} ({r[k+"_sd"]:.4f})' for k in MAIN])
    detection_rows=[]
    for d in order[:-1]:
        for m in ('xgboost','mlp'):
            g=table2.loc[(table2.dataset==d)&(table2.model==m)].set_index('method')
            detection_rows.append([NAMES[d],MODELS[m]]+[f'{g.loc[k,"mean"]:.3f}' for k in DETECT])
    report=['# Path A+ verified confirmation outcomes','', 'Five fitted-model seeds (11, 29, 47, 71, 101) on common data, explanation perturbations and monitor resamples. Cost is 5 for a false negative and 1 for a false positive. Lower cost AURC is better. Entries show mean (sample seed SD).','',md_table(['Benchmark','Model','Expected cost','Naive unscaled','Cost-preserved'],risk_rows),'','The interaction row uses the local conditional-boundary arm. Other controlled rows use fully shifted observations. Electricity uses the complete chronological deployment stream. The matched interaction covariate-only control is supplied separately.','', '## Table 2: window AUROC at W=100','',md_table(['Benchmark','Model','Scalar D','Signed A-B','A','B','Entropy','KS','MMD'],detection_rows),'','Higher window AUROC is better. Transition-overlapping windows are excluded. The interaction covariate-only and boundary arms have identical inputs and monitoring statistics. Source-calibrated thresholds and stationary alarm rates are in the accompanying tables.','',f'Independent audit: {audit["checks"]:,} passed checks, including every saved checkpoint and independently refitted XGBoost models.','', '## Paired outcome counts','']
    for subset,label in [(primary.loc[primary.dataset.isin(['sea','hyperplane','adult_numeric'])],'Original three controlled benchmark designs on fresh observations'),(primary.loc[primary.dataset=='interaction'],'Interaction boundary arm'),(primary,'All five benchmarks')]:
        for method in MAIN[1:]:
            delta=subset[method]-subset.expected_cost
            report.append(f'- {label}; {method}: {(delta<-1e-12).sum()} lower, {(delta>1e-12).sum()} higher, {(abs(delta)<=1e-12).sum()} tied AURCs.')
    report+=['',f'Source calibration reconstruction: {calibration_audit["checks"]:,} additional checks passed across every bootstrap replicate and source selection-window statistic.']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    provenance={'root_identity':audit['run_identity'],'engine_sha256':audit['engine_sha256'],'audit_checks':audit['checks'],'table1_seed_rows':len(primary),'table2_methods':list(DETECT),'scope':'Fresh confirmation cases, fixed model/split/explanation/monitor settings; all outcomes retained.'}
    provenance['source_calibration_checks']=calibration_audit['checks']
    provenance['total_confirmation_checks']=audit['checks']+calibration_audit['checks']
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    figures(root,out)
    print('\n'.join(report),flush=True)

def figures(root,out):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5})
    fig=plt.figure(figsize=(6.5,3.8));outer=fig.add_gridspec(1,2,width_ratios=[1,1.14],wspace=0.70)
    left=outer[0].subgridspec(2,1,hspace=0.64);right=outer[1].subgridspec(2,1,hspace=0.64)
    grid=np.linspace(.05,1,191);curve_records=[]
    panels=[('hyperplane','mlp','covariate_correlation_gradual'),('adult_numeric','xgboost','covariate_selection')]
    for k,(dataset,model,stream) in enumerate(panels):
        ax=fig.add_subplot(left[k])
        for method in MAIN:
            all_risk=[]
            for seed in (11,29,47,71,101):
                unit=root/'units'/dataset/model/f'seed_{seed}';r=read(unit/f'{stream}_rows.csv');meta=json.loads((unit/'stream_manifest.json').read_text())[stream]
                r=r.iloc[meta['transition_end']:];p=r.p.to_numpy();y=r.y.to_numpy();loss=np.where((y==1)&(p<.5),5.,np.where((y==0)&(p>=.5),1.,0.));cov,risk=curve(loss,r[f'risk_W100_{method}'].to_numpy())
                # Step-function interpolation matches tied-group AURC convention.
                values=risk[np.minimum(np.searchsorted(cov,grid,side='left'),len(risk)-1)]
                all_risk.append(values)
                curve_records.extend({'dataset':dataset,'model':model,'seed':seed,'method':method,'coverage':float(c),'risk':float(v)} for c,v in zip(grid,values))
            array=np.asarray(all_risk);mean=array.mean(0);sd=array.std(0,ddof=1)
            ax.plot(grid,mean,color=COLORS[method],label=LABELS[method],lw=1.65 if method=='expected_cost' else 1.05,ls='--' if method=='expected_cost' else '-')
            ax.fill_between(grid,np.maximum(0,mean-sd),mean+sd,color=COLORS[method],alpha=.10,lw=0)
        ax.set(xlim=(0.05,1),ylim=(0,None),ylabel='Accepted cost',xlabel='Coverage',title=('(a) ' if k==0 else '')+NAMES[dataset]+'/'+MODELS[model])
        if k==0:
            ax.legend(loc='upper left',frameon=False,ncol=3,columnspacing=.9,handlelength=1.3)
        ax.grid(alpha=.15,lw=.5)
    unit=root/'units/interaction/mlp/seed_11';r=read(unit/'local_boundary_shift_rows.csv');windows=read(unit/'monitor_windows.csv');w=windows.loc[(windows.stream=='local_boundary_shift')&(windows.window==100)&(windows.method=='disagreement')];cutoffs=read(unit/'selection_thresholds.csv')
    ax=fig.add_subplot(right[0]);ax.plot(w.end,w.statistic/w.threshold,color='#574e89',lw=1.1);ax.axhline(1,color='#777777',ls='--',lw=.8,label=r'$q_W$');ax.axvline(1200,color='#999999',ls=':',lw=.8)
    ax.set(xlim=(0,2400),ylabel=r'$\Delta_{t,W}/q_W$',xlabel='Stream sample',title='(b) Interaction/MLP, seed 11');ax.legend(loc='upper left',frameon=False)
    ax.grid(alpha=.15,lw=.5)
    ax2=fig.add_subplot(right[1]);p=r.p.to_numpy();y=r.y.to_numpy();loss=np.where((y==1)&(p<.5),5.,np.where((y==0)&(p>=.5),1.,0.));trace=[]
    for method in MAIN:
        threshold=cutoffs.loc[(cutoffs.window==100)&(cutoffs.method==method)&(cutoffs.review_budget==.1)].threshold.iloc[0]
        accepted=(r[f'risk_W100_{method}'].to_numpy()<=threshold).astype(float)
        numerator=pd.Series(loss*accepted).rolling(100,min_periods=100).sum();denominator=pd.Series(accepted).rolling(100,min_periods=100).sum()
        realized=numerator/denominator.replace(0,np.nan)
        ax2.plot(np.arange(len(r))+1,realized,color=COLORS[method],label=LABELS[method],lw=1.35 if method=='expected_cost' else .8,ls='--' if method=='expected_cost' else '-')
        trace.extend({'sample':i+1,'method':method,'rolling_accepted_cost':v,'rolling_coverage':denominator.iloc[i]/100,'threshold':threshold} for i,v in enumerate(realized))
    ax2.axvline(1200,color='#999999',ls=':',lw=.8);ax2.set(xlim=(0,2400),ylim=(0,None),ylabel='Accepted cost',xlabel='Stream sample');ax2.grid(alpha=.15,lw=.5)
    fig.subplots_adjust(left=.085,right=.99,bottom=.10,top=.94)
    fig.savefig(figdir/'figure1.pdf',dpi=600);fig.savefig(figdir/'figure1.png',dpi=600);plt.close(fig)
    pd.DataFrame(curve_records).to_csv(figdir/'panel_a_source.csv',index=False);pd.DataFrame(trace).to_csv(figdir/'panel_b_source.csv',index=False)
    (figdir/'figure1_design.json').write_text(json.dumps({'panel_a':'Hyperplane/MLP and Adult/XGB: five-seed mean risk and sample SD, fully shifted cases, fixed coverage grid 0.05 to 1; all three methods share predictions and labels.','panel_b':'Prespecified seed 11, interaction/MLP local-boundary arm, W100 and trailing accepted cost of 100 observations at source 10% review thresholds; vertical line is onset1200.','horizontal_subplot_spacing':0.70,'vector_pdf':True,'dpi_png':600,'pdf_fonttype':42,'selection_of_example':'First prespecified temporal-example seed; Adult/XGB replaces Adult/MLP for curve separation in the final presentation pass.','model_retraining':False},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args();analyze(a.root,a.out)
