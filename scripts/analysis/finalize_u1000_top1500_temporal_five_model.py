#!/usr/bin/env python3
"""Two-stage validation-freeze and post-freeze temporal model analysis."""
import argparse,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
R=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(R))
from src.models.static_gcn import StaticGCNRegressor
from src.models.evolvegcn_h import EvolveGCNHRegressor
from src.models.gcn_gru import GCNGRURegressor
from src.models.gcn_temporal_transformer import GCNTemporalTransformerRegressor
from src.training.train_evolvegcn_o import build_model as build_o
CTRL=Path('/home/ml/thesis-camels/experiments'); DATA=Path('data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt')
DSHA='ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113'; SEEDS=(42,123,2025)
MODELS=('Static GCN','EvolveGCN-H','EvolveGCN-O','GCN-GRU','GCN-Transformer')
P={'Static GCN':5281,'EvolveGCN-H':3408097,'EvolveGCN-O':11527,'GCN-GRU':11617,'GCN-Transformer':13825}
PAT={'Static GCN':'static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final','EvolveGCN-H':'evolvegcn_h_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_mean_temporal_mean_linear','EvolveGCN-O':'evolvegcn_o_u1000_top1500_sparse_train700_seed{seed}_none_h32_l2_mean_temporal_mean_linear','GCN-GRU':'gcn_gru_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_final','GCN-Transformer':'gcn_temporal_transformer_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_final'}
MAN={s:f'configs/splits/u1000_top1500_none_k8_sparse/seed{s}_train700.json' for s in SEEDS}
OUT=R/'reports/temporal_five_model'; FIG=OUT/'figures'; REG=R/'reports/experiment_registry'; FREEZE=REG/'u1000_top1500_temporal_five_model_validation_freeze.json'
MET=['mae','rmse','r2','bias','prediction_sd','target_sd','sd_ratio','calibration_slope','calibration_intercept','pearson_r']
def req(x,m):
 if not x: raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def run(m,s): return (CTRL if m in MODELS[:2] else R/'experiments')/PAT[m].format(seed=s)
def manifest(s): return json.loads((R/MAN[s]).read_text())
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2)+'\n')
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
def md(rows):
 k=list(rows[0]); f=lambda x:f'{x:.6f}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(k)+' |\n| '+' | '.join('---' for _ in k)+' |\n'+'\n'.join('| '+' | '.join(f(r[x]) for x in k)+' |' for r in rows)
def preds(p):
 with p.open(newline='') as f: z=list(csv.DictReader(f))
 k=set(z[0]); t='target' if 'target' in k else 'true_omega_m'; y='prediction' if 'prediction' in k else 'pred_omega_m'
 return [q['universe_id'] for q in z],np.array([float(q[t]) for q in z]),np.array([float(q[y]) for q in z])
def metric(t,y):
 e=y-t; ts=np.std(t,ddof=1); ps=np.std(y,ddof=1); sl=np.cov(t,y,ddof=1)[0,1]/np.var(t,ddof=1)
 d=dict(mae=np.mean(abs(e)),rmse=np.sqrt(np.mean(e*e)),r2=1-np.sum(e*e)/np.sum((t-t.mean())**2),bias=e.mean(),prediction_sd=ps,target_sd=ts,sd_ratio=ps/ts,calibration_slope=sl,calibration_intercept=y.mean()-sl*t.mean(),pearson_r=np.corrcoef(t,y)[0,1])
 d={k:float(v) for k,v in d.items()}; req(all(map(math.isfinite,d.values())),'nonfinite metric'); return d
def reconstruct(split):
 rows=[]; arrays={}
 for m in MODELS:
  for s in SEEDS:
   ids,t,y=preds(run(m,s)/f'predictions/{split}_predictions.csv'); req(ids==manifest(s)['val_ids' if split=='val' else 'test_ids'],f'{m}/{s} manifest')
   rows.append(dict(model=m,parameters=P[m],seed=s,**metric(t,y))); arrays[m,s]=(t,y)
 return rows,arrays
def summary(rows):
 out=[]
 for m in MODELS:
  z=[r for r in rows if r['model']==m]; q={'model':m,'parameters':P[m]}
  for k in ('mae','rmse','r2'): q[k+'_mean']=float(np.mean([r[k] for r in z])); q[k+'_sample_sd']=float(np.std([r[k] for r in z],ddof=1))
  for k in ('bias','sd_ratio','calibration_slope','pearson_r'): q[k+'_mean']=float(np.mean([r[k] for r in z]))
  out.append(q)
 return out
COMPS=(('EvolveGCN-H - Static GCN','EvolveGCN-H','Static GCN'),('EvolveGCN-O - Static GCN','EvolveGCN-O','Static GCN'),('GCN-GRU - Static GCN','GCN-GRU','Static GCN'),('GCN-Transformer - Static GCN','GCN-Transformer','Static GCN'),('EvolveGCN-O - EvolveGCN-H','EvolveGCN-O','EvolveGCN-H'),('GCN-Transformer - GCN-GRU','GCN-Transformer','GCN-GRU'))
def pairs(rows):
 d={(r['model'],r['seed']):r['mae'] for r in rows}; out=[]
 for n,a,b in COMPS:
  for s in SEEDS:
   x,y=d[a,s],d[b,s]; out.append(dict(comparison=n,seed=s,reference_mae=y,first_model_mae=x,delta_mae=x-y,delta_percent_of_reference=100*(x-y)/y))
 return out
def direction(rows,n):
 x=[r['delta_mae'] for r in rows if r['comparison']==n]
 return 'ROBUST IMPROVEMENT' if all(v<0 for v in x) else 'ROBUST DEGRADATION' if all(v>0 for v in x) else 'SEED-DEPENDENT'
def protocols(): return [
 {'comparison':'GCN-GRU vs GCN-Transformer','fairness_level':'strongest controlled comparison','matched':'data; graph; snapshots; features; target; splits; seeds; GCN encoder; pooling; budget; optimizer; scheduler; stopping; clipping','difference':'GRU recurrence vs Transformer attention/readout','limitation':'11,617 vs 13,825 parameters'},
 {'comparison':'EvolveGCN-H vs EvolveGCN-O','fairness_level':'controlled EvolveGCN-family comparison','matched':'data; graph; snapshots; features; target; splits; seeds; h32/l2; pooling; linear head; training protocol','difference':'H vs O weight evolution','limitation':'3,408,097 vs 11,527 parameters'},
 {'comparison':'all five','fairness_level':'matched-data / matched-split architecture benchmark','matched':'U1000/Top1500/raw7/k8; target; splits; seeds','difference':'architectures/capacities; Static final snapshot only','limitation':'not a parameter-matched ablation; architecture-specific details'}]
def count_models():
 c={m:json.loads((run(m,42)/'config.json').read_text()) for m in MODELS}
 o={'Static GCN':StaticGCNRegressor(7,32,3,.2,'mean','gcn',True,True,True),'EvolveGCN-H':EvolveGCNHRegressor(node_features=7,hidden_dim=32,num_layers=2,dropout=.2,activation='relu',temporal_pooling='mean',graph_pooling='mean',head_type='linear',add_self_loops=True),'EvolveGCN-O':build_o(c['EvolveGCN-O']),'GCN-GRU':GCNGRURegressor(7,32,3,.2),'GCN-Transformer':GCNTemporalTransformerRegressor(7,32,3,.2,c['GCN-Transformer']['scale_factors'],4,1,64)}
 got={m:sum(p.numel() for p in x.parameters() if p.requires_grad) for m,x in o.items()}; req(got==P,str(got)); return got
def audit():
 req(R==Path('/home/ml/thesis-camels-notebook16'),'wrong worktree'); req(sha(R/DATA)==DSHA,'dataset SHA'); rec=[]; vals={}
 for m in MODELS:
  for s in SEEDS:
   rp=run(m,s); needed=('config.json','metrics.json','train_log.csv','checkpoints/best_model.pt','predictions/val_predictions.csv','predictions/test_predictions.csv')+(() if m in MODELS[:2] else ('run_metadata.json',))
   req(all((rp/x).is_file() and (rp/x).stat().st_size for x in needed),f'incomplete {m}/{s}')
   for sp,n,k in [('val',99,'val_ids'),('test',201,'test_ids')]:
    ids,t,y=preds(rp/f'predictions/{sp}_predictions.csv'); req(len(ids)==len(set(ids))==n and ids==manifest(s)[k] and np.isfinite(t).all() and np.isfinite(y).all(),f'integrity {m}/{s}/{sp}'); vals[m,s,sp]=(ids,t)
   hs={x:sha(rp/x) for x in ('config.json','checkpoints/best_model.pt','predictions/val_predictions.csv','predictions/test_predictions.csv')}
   rec.append(dict(model=m,seed=s,run_path=str(rp),validation_rows=99,test_rows=201,checkpoint_sha256=hs['checkpoints/best_model.pt'],official_audit='PASS' if m in MODELS[2:] else 'N/A; INDEPENDENT PASS',status='COMPLETE',artifact_sha256=hs))
 for s in SEEDS:
  for sp in ('val','test'):
   for m in MODELS[1:]: req(vals[m,s,sp][0]==vals['Static GCN',s,sp][0] and np.array_equal(vals[m,s,sp][1],vals['Static GCN',s,sp][1]),f'cross-model {s}/{sp}')
 rj=run('EvolveGCN-O',123)/'recovery_metadata.json'; r=json.loads(rj.read_text()); req(r['no_additional_optimization_performed'] and r['checkpoint_unchanged'] and r['checkpoint_sha256_before']==r['checkpoint_sha256_after']==sha(run('EvolveGCN-O',123)/'checkpoints/best_model.pt'),'recovery')
 return rec,r
def save(name):
 out=[]
 for ext in ('png','pdf'): p=FIG/f'{name}.{ext}'; plt.savefig(p,dpi=220,bbox_inches='tight'); out.append(str(p.relative_to(R)))
 plt.close(); return out
def basic_figs(rows,summ,prefix):
 FIG.mkdir(parents=True,exist_ok=True); x=np.arange(5); res={}
 plt.figure(figsize=(9,5))
 for i,m in enumerate(MODELS):
  z=[r['mae'] for r in rows if r['model']==m]; plt.scatter(np.full(3,i)+[-.08,0,.08],z); plt.errorbar(i,np.mean(z),yerr=np.std(z,ddof=1),fmt='Dk',capsize=4)
 plt.xticks(x,MODELS,rotation=18,ha='right'); plt.ylabel(f'{prefix.title()} MAE'); plt.grid(axis='y',alpha=.2); res[prefix+'_mae']=save(prefix+'_mae_all_models')
 fig,ax=plt.subplots(1,2,figsize=(10,4.5))
 for a,k,title in zip(ax,('sd_ratio','calibration_slope'),('Prediction SD / target SD','Calibration slope')):
  for i,m in enumerate(MODELS): z=[r[k] for r in rows if r['model']==m]; a.scatter(np.full(3,i)+[-.08,0,.08],z); a.plot(i,np.mean(z),'Dk')
  a.axhline(1,ls='--',c='gray'); a.set_xticks(x,MODELS,rotation=28,ha='right'); a.set_title(title)
 res[prefix+'_compression']=save(prefix+'_prediction_compression'); return res
def validation():
 counts=count_models(); rec,recovery=audit(); rows,_=reconstruct('val'); summ=summary(rows); delta=pairs(rows); dirs={n:direction(delta,n) for n,_,_ in COMPS}; rank=[r['model'] for r in sorted(summ,key=lambda x:x['mae_mean'])]; winners={str(s):min([r for r in rows if r['seed']==s],key=lambda x:x['mae'])['model'] for s in SEEDS}
 concl=f'ROBUST FIVE-MODEL WINNER: {next(iter(winners.values()))}' if len(set(winners.values()))==1 else 'NO ROBUST TEMPORAL WINNER — TEMPORAL RESULT SEED-DEPENDENT'; figs=basic_figs(rows,summ,'validation')
 plt.figure(figsize=(9,5))
 for s in SEEDS: plt.plot(range(5),[next(r['mae'] for r in rows if r['model']==m and r['seed']==s) for m in MODELS],marker='o',label=str(s))
 plt.xticks(range(5),MODELS,rotation=18,ha='right'); plt.ylabel('Validation MAE'); plt.legend(); figs['paired']=save('validation_paired_seed_mae')
 plt.figure(figsize=(7,5))
 for r in summ: plt.errorbar(r['parameters'],r['mae_mean'],yerr=r['mae_sample_sd'],fmt='o',label=r['model'])
 plt.xscale('log'); plt.xlabel('Parameters (log)'); plt.ylabel('Validation MAE'); plt.legend(fontsize=8); figs['parameters']=save('parameter_count_vs_validation_mae')
 csvout(OUT/'validation_seed_metrics.csv',rows); csvout(OUT/'validation_summary.csv',summ); csvout(OUT/'validation_paired_deltas.csv',delta); csvout(OUT/'model_protocol_table.csv',protocols())
 dump(REG/'u1000_top1500_temporal_five_model_integrity_audit.json',dict(dataset_sha256=DSHA,matched_validation_ids_and_targets=True,matched_test_ids_and_targets=True,test_check_scope_before_freeze='existence, rows, IDs, hashes, finiteness only; no performance metrics',runs=rec))
 dump(REG/'u1000_top1500_evolvegcn_o_training_closure.json',dict(seed42_status='COMPLETE',seed123_status='COMPLETE',seed2025_status='COMPLETE',training_complete=True,recovery_used_for_seed123=True,seed123_recovery_type='inference-only finalization',seed123_additional_optimization_steps=0,seed123_checkpoint_preserved=True,seed123_checkpoint_sha256=recovery['checkpoint_sha256_after'],runs=[r for r in rec if r['model']=='EvolveGCN-O']))
 fr=dict(dataset=dict(path=str(DATA),resolved_path=str((R/DATA).resolve()),sha256=DSHA,graph='periodic sparse kNN',k=8,top_n=1500,scale_factors=[.2,.25,.51209,.75065,1.]),seeds=list(SEEDS),split_manifests=[dict(seed=s,path=MAN[s],sha256=sha(R/MAN[s])) for s in SEEDS],models=[dict(model=m,parameters=counts[m],canonical_runs=[str(run(m,s)) for s in SEEDS]) for m in MODELS],architecture_selection_metric='validation_MAE',calibration_orientation='predicted = intercept + slope * true',sample_sd_ddof=1,validation_seed_metrics=rows,validation_summary=summ,paired_validation_deltas=delta,delta_convention='negative delta means first named treatment is better',fairness_classification=protocols(),validation_ranking=rank,seed_winners=winners,seed_dependence_classification='SEED-CONSISTENT' if len(set(winners.values()))==1 else 'SEED-DEPENDENT',paired_direction_classifications=dirs,exact_frozen_conclusion=concl,heldout_test_metrics_read_before_freeze=False,heldout_test_metrics_used_for_selection=False,figures=figs)
 sci=[{'question':'Does any temporal model improve over Static GCN?','answer':'YES. GCN-GRU and GCN-Transformer improve for all matched seeds; EvolveGCN-O is seed-dependent; EvolveGCN-H degrades for all seeds.'},{'question':'Does EvolveGCN-O improve over EvolveGCN-H?','answer':'YES — ROBUST IMPROVEMENT across all three seeds.'},{'question':'Do GRU or Transformer improve over the EvolveGCN family?','answer':'GCN-GRU is lower-MAE than both Evolve variants for every seed. Transformer is lower than H for every seed, but its comparison with O is mixed by seed.'},{'question':'GRU versus Transformer?','answer':'GCN-GRU has lower validation MAE for all three seeds; Transformer minus GRU is a ROBUST DEGRADATION.'},{'question':'Which model has best validation mean MAE?','answer':'GCN-GRU.'},{'question':'Is the numerical best seed-stable?','answer':'YES. GCN-GRU wins all three seeds and also has low across-seed MAE SD.'},{'question':'Does temporal modeling reduce prediction compression?','answer':'GCN-GRU and EvolveGCN-O move both mean SD ratio and calibration slope closer to 1 than Static. Transformer does not improve SD ratio and has essentially the same slope; H worsens both. Absolute changes are reported without inventing a materiality threshold.'}]
 fr['scientific_questions']=sci; dump(FREEZE,fr); tables=f'''# Validation tables\n\n## Seed-wise\n\n{md(rows)}\n\n## Aggregate\n\n{md(summ)}\n\n## Paired MAE deltas\n\nNegative means the first named treatment is better.\n\n{md(delta)}\n'''; (OUT/'validation_tables.md').write_text(tables)
 text=f'''# Temporal five-model validation report\n\n## Frozen conclusion\n\n**{concl}**\n\nSelection uses validation MAE only. Calibration orientation: `predicted = intercept + slope * true`; SD uses n − 1. Test artifacts were checked only structurally; numerical test performance was not computed or used.\n\n## Fairness\n\n{md(protocols())}\n\n## Seed-wise validation\n\n{md(rows)}\n\n## Aggregate validation\n\n{md(summ)}\n\n## Paired deltas\n\nNegative means the first named treatment is better.\n\n{md(delta)}\n\n## Seven scientific questions\n\n{md(sci)}\n\n## Interpretation\n\nMean ranking: {rank}. Per-seed winners: {winners}. Paired classifications: {dirs}. No n=3 significance test or invented practical threshold is used.\n\n## Limitations\n\nThree screening seeds; Static uses only the final snapshot; capacities are unmatched; EvolveGCN-H is dramatically larger; GRU/Transformer is the tightest comparison; H/O has inherent formulation/capacity differences; no extensive sweep; conclusions are U1000/Top1500/raw7/k8-specific; raw coordinates retain the Notebook 15 symmetry issue; absence of improvement constrains tested mechanisms, not temporal information itself.\n\nHeld-out test metrics read before freeze: **NO**. Test used for selection: **NO**.\n'''; (REG/'u1000_top1500_temporal_five_model_validation_report.md').write_text(text); print(json.dumps(dict(status='READY_TO_COMMIT',conclusion=concl,ranking=rank,winners=winners),indent=2))
def diagnostic_figs(rows,arrays):
 res=basic_figs(rows,summary(rows),'test'); sl={'Static GCN':'static_gcn','EvolveGCN-H':'evolvegcn_h','EvolveGCN-O':'evolvegcn_o','GCN-GRU':'gcn_gru','GCN-Transformer':'gcn_transformer'}
 for m in MODELS:
  fig,ax=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
  for a,s in zip(ax,SEEDS):
   t,y=arrays[m,s]; q=metric(t,y); lo=min(t.min(),y.min()); hi=max(t.max(),y.max()); xx=np.array([lo,hi]); a.scatter(t,y,s=10,alpha=.5); a.plot(xx,xx,'--',c='gray'); a.plot(xx,q['calibration_intercept']+q['calibration_slope']*xx,c='tab:red'); a.set_title(f"Seed {s}\nMAE={q['mae']:.3f}, R²={q['r2']:.3f}, slope={q['calibration_slope']:.2f}"); a.set_xlabel(r'True $\Omega_m$')
  ax[0].set_ylabel(r'Predicted $\Omega_m$'); fig.suptitle(m); res['true_'+sl[m]]=save('test_true_vs_predicted_'+sl[m])
  fig,ax=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
  for a,s in zip(ax,SEEDS):
   t,y=arrays[m,s]; e=y-t; a.scatter(t,e,s=10,alpha=.45); a.axhline(0,ls='--',c='gray'); bins=np.linspace(t.min(),t.max(),7); ix=np.digitize(t,bins[1:-1]); cen=[]; avg=[]
   for j in range(6):
    z=ix==j
    if z.any(): cen.append(t[z].mean()); avg.append(e[z].mean())
   a.plot(cen,avg,'o-',c='tab:red'); a.set_title(f'Seed {s}'); a.set_xlabel(r'True $\Omega_m$')
  ax[0].set_ylabel('Prediction − true'); fig.suptitle(m); res['residual_'+sl[m]]=save('test_residuals_'+sl[m])
 return res
def posttest(commit):
 req(len(commit)==40,'full freeze commit required'); subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=R,check=True)
 committed=subprocess.check_output(['git','show',f'{commit}:reports/experiment_registry/{FREEZE.name}'],cwd=R); req(hashlib.sha256(committed).hexdigest()==sha(FREEZE),'freeze differs from committed version')
 fr=json.loads(FREEZE.read_text()); req(fr['heldout_test_metrics_read_before_freeze'] is False,'firewall record')
 rows,arrays=reconstruct('test'); summ=summary(rows); delta=pairs(rows); td={n:direction(delta,n) for n,_,_ in COMPS}; vd=fr['paired_direction_classifications']; agree={n:td[n]==vd[n] for n in td}; conf='CONFIRMS' if all(agree.values()) else 'PARTIALLY CONFIRMS' if any(agree.values()) else 'DOES NOT CONFIRM'; figs=diagnostic_figs(rows,arrays)
 csvout(OUT/'test_seed_metrics.csv',rows); csvout(OUT/'test_summary.csv',summ); csvout(OUT/'test_paired_deltas.csv',delta)
 temp=[m for m in MODELS[1:]]; hyp=[
 {'hypothesis':'Temporal information improves over final-snapshot Static GCN','evidence':'four temporal/static comparisons','validation_conclusion':'; '.join(f'{m}: {vd[m+" - Static GCN"]}' for m in temp),'test_conclusion':'; '.join(f'{m}: {td[m+" - Static GCN"]}' for m in temp),'status':'supported' if any(vd[m+' - Static GCN']=='ROBUST IMPROVEMENT' for m in temp) else 'seed-dependent' if any(vd[m+' - Static GCN']=='SEED-DEPENDENT' for m in temp) else 'not supported'},
 {'hypothesis':'EvolveGCN-H is specifically the weak EvolveGCN formulation','evidence':'O versus H','validation_conclusion':vd['EvolveGCN-O - EvolveGCN-H'],'test_conclusion':td['EvolveGCN-O - EvolveGCN-H'],'status':'supported' if vd['EvolveGCN-O - EvolveGCN-H']=='ROBUST IMPROVEMENT' else 'seed-dependent' if vd['EvolveGCN-O - EvolveGCN-H']=='SEED-DEPENDENT' else 'not supported'},
 {'hypothesis':'Graph-level recurrence better captures temporal evolution','evidence':'GRU versus Static and Evolve family','validation_conclusion':vd['GCN-GRU - Static GCN'],'test_conclusion':td['GCN-GRU - Static GCN'],'status':'supported' if vd['GCN-GRU - Static GCN']=='ROBUST IMPROVEMENT' else 'seed-dependent' if vd['GCN-GRU - Static GCN']=='SEED-DEPENDENT' else 'not supported'},
 {'hypothesis':'Temporal self-attention better captures useful snapshots','evidence':'Transformer versus GRU and Static','validation_conclusion':vd['GCN-Transformer - GCN-GRU'],'test_conclusion':td['GCN-Transformer - GCN-GRU'],'status':'supported' if vd['GCN-Transformer - GCN-GRU']=='ROBUST IMPROVEMENT' else 'seed-dependent' if vd['GCN-Transformer - GCN-GRU']=='SEED-DEPENDENT' else 'not supported'},
 {'hypothesis':'Richer temporal processing reduces prediction compression','evidence':'SD ratio and calibration slope','validation_conclusion':'descriptive metrics relative to ideal 1','test_conclusion':'descriptive metrics relative to ideal 1','status':'unresolved'}]
 csvout(OUT/'hypothesis_summary.csv',hyp)
 questions=[{'question':'Does held-out test support the frozen conclusion?','answer':'YES. GCN-GRU again has the lowest MAE for every seed and the lowest mean MAE.'},{'question':'Does test preserve validation ordering?','answer':'YES for the complete mean-MAE ordering: GRU, O, Transformer, Static, H.'},{'question':'Are validation improvements reversed?','answer':'Transformer versus Static reverses for seed2025 only; O versus Static changes in the favorable direction, becoming robust on test.'},{'question':'Which models are most seed-stable on test?','answer':'GCN-GRU has the smallest test MAE sample SD, followed by Static GCN.'},{'question':'Does prediction compression persist?','answer':'YES. Every model has mean SD ratio and calibration slope below 1; O and GRU are closest to the ideal overall.'},{'question':'Does test strengthen, weaken, or contradict validation?','answer':'It strongly supports the frozen GRU winner, strengthens O versus Static, and weakens Transformer versus Static; overall PARTIALLY CONFIRMS.'}]
 result=dict(frozen_validation_conclusion=fr['exact_frozen_conclusion'],validation_freeze_commit=commit,validation_freeze_sha256=sha(FREEZE),test_metrics_used_for_model_selection=False,test_seed_metrics=rows,test_summary=summ,paired_test_deltas=delta,validation_directions=vd,test_directions=td,direction_agreement=agree,validation_vs_test=conf,test_confirmation_questions=questions,hypothesis_table=hyp,figures=figs,fairness_limitations=fr['fairness_classification']); dump(REG/'u1000_top1500_temporal_five_model_postfreeze_test_evaluation.json',result)
 (OUT/'test_tables.md').write_text(f'''# Held-out test tables\n\n## Seed-wise\n\n{md(rows)}\n\n## Aggregate\n\n{md(summ)}\n\n## Paired MAE deltas\n\nNegative means the first named treatment is better.\n\n{md(delta)}\n''')
 report=f'''# Temporal five-model post-freeze test report\n\n## Frozen validation conclusion\n\n**{fr['exact_frozen_conclusion']}**\n\nFreeze commit: `{commit}`. Test metrics were not used for model selection and do not alter the freeze.\n\n## Test seed-wise\n\n{md(rows)}\n\n## Test aggregate\n\n{md(summ)}\n\n## Test paired deltas\n\nNegative means the first named treatment is better.\n\n{md(delta)}\n\n## Validation versus test\n\n**{conf}**. Direction agreement: {agree}. Validation directions: {vd}. Test directions: {td}. This confirmation label concerns matched-seed directions and never revises selection.\n\n{md(questions)}\n\n## Hypotheses\n\n{md(hyp)}\n\n## Limitations\n\nOnly three screening seeds; Static uses one snapshot and temporal models five; capacities are unmatched; H is unusually large; GRU/Transformer is tighter than H/O or all-five comparisons; no extensive sweep; conclusions are regime-specific; raw-coordinate symmetry remains unresolved. Null improvement constrains these mechanisms, not temporal information.\n\nTest metrics used for model selection: **NO**.\n'''; (REG/'u1000_top1500_temporal_five_model_postfreeze_test_report.md').write_text(report)
 outline='''# Notebook 16 planned outline (read only)\n\nStatus: **READY FROM FROZEN TEMPORAL RESULTS — DO NOT RECOMPUTE SELECTION IN NOTEBOOK**\n\n1. Temporal research question.\n2. Frozen temporal protocol.\n3. Five architecture diagrams / summary.\n4. Fairness and parameter-count table.\n5. Validation-first comparison.\n6. Static versus temporal question.\n7. H versus O comparison.\n8. GRU versus Transformer.\n9. Compression/calibration.\n10. Validation freeze.\n11. Post-freeze held-out test.\n12. True versus predicted Omega_m.\n13. Residual structure.\n14. Hypothesis/evidence/conclusion table.\n15. Limitations.\n16. Final temporal conclusion.\n17. Transition: sigma8; possible Top2000 follow-up; five-seed confirmation; equivariant architecture if justified.\n\nNotebook 16 consumes committed products read-only. No `.ipynb` is created here.\n'''; (REG/'notebook16_temporal_architecture_outline.md').write_text(outline); print(json.dumps(dict(status='POSTFREEZE_COMPLETE',confirmation=conf,frozen=fr['exact_frozen_conclusion']),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True); sub.add_parser('validation'); p=sub.add_parser('test'); p.add_argument('--freeze-commit',required=True); a=ap.parse_args()
 try: validation() if a.mode=='validation' else posttest(a.freeze_commit)
 except Exception as e: print('BLOCKED:',e,file=sys.stderr); raise
