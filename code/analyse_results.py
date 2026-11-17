"""Generate all reported quantities from frozen final outputs, without retuning."""
from __future__ import annotations
import json,platform,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
import revision_core as c
from run_revision import RESULTS,error,bundles

LABELS={'tall_buildings':'High rise','horizontal_city':'Horizontal low rise','small_town':'Small town','suburban':'Suburban','delft_3dbag':'3DBAG tile','rotterdam_delfshaven':'Rotterdam Delfshaven'}

def load_all(require_complete=True):
    frames=[];receivers=[]
    for path in sorted((RESULTS/'validation').glob('*/*/grid.npz')):
        z=np.load(path);meta=json.loads((path.parent/'metadata.json').read_text());v=z['values']
        ident=path.parent.parent.name+'/'+path.parent.name
        df=pd.DataFrame(v,columns=['raw','proxy','radius','topk','exact_marginal_union','float','packed'])
        df['az']=z['az'];df['el']=z['el'];df['overlap_excess']=z['overlap'];df['multiplicity']=z['multiplicity'];df['single_max']=z['single_max'];df['id']=ident
        for k in ['scene','terrain','split','label','area_m2','retained_casters','candidate_casters']:df[k]=meta[k]
        df['kind']='wall' if meta['kind']=='wall' else 'roof';df['pitch_deg']=np.degrees(np.arccos(np.clip(meta['normal'][2],-1,1)))
        df['incidence']=c.directions(z['az'],z['el'])@np.asarray(meta['normal']);df['signed_error']=df.packed-df.raw;df['abs_error']=df.signed_error.abs()
        # A fixed local angular-neighbour contrast, not fitted to error outcomes.
        aidx=np.rint((z['az']-2.5)/10).astype(int);eidx=np.rint((z['el']-2.5)/5).astype(int)
        matrix=np.full((18,36),np.nan);matrix[eidx,aidx]=v[:,0];transition=np.zeros(len(v))
        for i,(yy,xx) in enumerate(zip(eidx,aidx)):
            neighbours=[matrix[yy,(xx-1)%36],matrix[yy,(xx+1)%36]]
            if yy>0:neighbours.append(matrix[yy-1,xx])
            if yy<17:neighbours.append(matrix[yy+1,xx])
            finite=np.array(neighbours)[np.isfinite(neighbours)]
            transition[i]=np.max(np.abs(finite-v[i,0])) if len(finite) else 0
        df['transition_contrast']=transition;frames.append(df)
        receivers.append(dict(id=ident,**meta,**error(v[:,0],v[:,6])))
    if require_complete:assert len(receivers)==256,('incomplete final panel',len(receivers))
    return pd.concat(frames,ignore_index=True),pd.DataFrame(receivers)

def metrics_table(df,group_fields):
    rows=[]
    for key,sub in df.groupby(group_fields,sort=False):
        if not isinstance(key,tuple):key=(key,)
        rows.append(dict(zip(group_fields,key),receivers=sub.id.nunique(),**error(sub.raw.to_numpy(),sub.packed.to_numpy())))
    return rows

def bootstrap(df,replicates=10000):
    # Stratified scene resampling for synthetic forms; fixed external patches.
    x=df.assign(active=df.raw>.01,weighted_abs=np.where(df.raw>.01,df.abs_error,0))
    rec=x.groupby(['terrain','scene','id'],sort=False).agg(s=('weighted_abs','sum'),n=('active','sum')).reset_index()
    rng=np.random.default_rng(610225);draw=np.empty(replicates)
    grouped={t:{s:g[['s','n']].to_numpy() for s,g in tg.groupby('scene')} for t,tg in rec.groupby('terrain')}
    for i in range(replicates):
        total=np.zeros(2)
        for terrain,scenes in grouped.items():
            keys=list(scenes)
            drawn=rng.choice(keys,size=len(keys),replace=True) if len(keys)>1 else keys
            for key in drawn:
                arr=scenes[key];idx=rng.integers(0,len(arr),len(arr));total+=arr[idx].sum(axis=0)
        draw[i]=total[0]/total[1] if total[1]>0 else np.nan
    return dict(low=float(np.nanpercentile(draw,2.5)),high=float(np.nanpercentile(draw,97.5)),replicates=replicates,seed=610225,interpretation='conditional on the sampled morphology strata and the two fixed external patches')

def analysis():
    df,rec=load_all();out=RESULTS/'analysis';out.mkdir(exist_ok=True)
    c.old.atomic_csv(out/'receiver_metrics.csv',rec.to_dict('records'))
    df.to_csv(out/'directional_results.csv.gz',index=False,compression='gzip')
    grouped=metrics_table(df,['terrain']);c.old.atomic_csv(out/'group_metrics.csv',grouped)
    kinds=metrics_table(df,['terrain','kind']);c.old.atomic_csv(out/'kind_metrics.csv',kinds)
    stages=[]
    comparisons=[('Proxy geometry','raw','proxy'),('Radius','proxy','radius'),('Rank limit','radius','topk'),('Aggregation of exact marginal fractions','topk','exact_marginal_union'),('Approximation through fixed aggregation','exact_marginal_union','float'),('Combined representation','topk','float'),('Quantisation','float','packed'),('Combined final','raw','packed')]
    for split,sub in [('synthetic',df[df.split=='validation']),('external',df[df.split=='external'])]:
        for name,ref,est in comparisons:
            if split=='external' and name=='Proxy geometry':continue
            stages.append(dict(split=split,stage=name,**error(sub[ref].to_numpy(),sub[est].to_numpy())))
    c.old.atomic_csv(out/'stage_metrics.csv',stages)
    diagnostics=[]
    conditions={
      'elevation_under_10':df.el<10,'elevation_10_to_30':(df.el>=10)&(df.el<30),'elevation_30_to_60':(df.el>=30)&(df.el<60),'elevation_60_plus':df.el>=60,
      'grazing_incidence_under_0.2':df.incidence<.2,'non_grazing_incidence':df.incidence>=.2,
      'sharp_neighbour_contrast_above_0.25':df.transition_contrast>.25,'lower_neighbour_contrast':df.transition_contrast<=.25,
      'one_active_caster':df.multiplicity==1,'two_to_five_active_casters':df.multiplicity.between(2,5),'six_or_more_active_casters':df.multiplicity>=6,
      'walls':df.kind=='wall','flat_roofs':(df.kind=='roof')&(df.pitch_deg<5),'pitched_roofs':(df.kind=='roof')&(df.pitch_deg>=5)}
    for name,mask in conditions.items():
        sub=df[mask];diagnostics.append(dict(condition=name,**error(sub.raw.to_numpy(),sub.packed.to_numpy()),fraction_error_over_025=float((sub.abs_error>.25).mean()),fraction_error_over_050=float((sub.abs_error>.5).mean())))
    c.old.atomic_csv(out/'error_conditions.csv',diagnostics)
    worst=df.sort_values('abs_error',ascending=False).head(20).to_dict('records');c.old.atomic_json(out/'worst_cases.json',dict(cases=worst))
    calibration=pd.read_csv(RESULTS/'kernel_gamma_calibration.csv');settings=json.loads((RESULTS/'calibration_final.json').read_text())
    kernelbest=[]
    for kernel,kd in calibration.groupby('kernel'):
        curve=kd.groupby('gamma')[['active_mae','active_p95']].mean();g=float(curve.active_mae.idxmin())
        kernelbest.append(dict(kernel=kernel,gamma=g,mae=float(curve.loc[g,'active_mae']),p95_mean=float(curve.loc[g,'active_p95'])))
    c.old.atomic_csv(out/'kernel_comparison.csv',kernelbest)
    gamma_rows=[]
    selected=calibration[calibration.kernel==settings['kernel']]
    for grouping in [['terrain'],['kind'],['terrain','kind']]:
        for key,sub in selected.groupby(grouping):
            if not isinstance(key,tuple):key=(key,)
            curve=sub.groupby('gamma').active_mae.mean();best=float(curve.idxmin());fixed=float(curve.loc[settings['gamma']])
            gamma_rows.append(dict(group='/'.join(key),optimal_calibration_gamma=best,optimal_mae=float(curve.min()),global_gamma_mae=fixed,global_minus_group_mae=fixed-float(curve.min()),**{f'mae_gamma_{g:.1f}':float(v) for g,v in curve.items()}))
    c.old.atomic_csv(out/'gamma_sensitivity.csv',gamma_rows)
    audit=pd.read_csv(RESULTS/'original_reference_audit.csv')
    correction=dict(n=len(audit),changed_raw_above_1e6=int((np.abs(audit.old_raw-audit.corrected_raw)>1e-6).sum()),changed_proxy_above_1e6=int((np.abs(audit.old_proxy-audit.corrected_proxy)>1e-6).sum()),max_raw_change=float(np.abs(audit.old_raw-audit.corrected_raw).max()),raw_change_mae=float(np.abs(audit.old_raw-audit.corrected_raw).mean()),old_proxy_failure_correction=json.loads((RESULTS/'original_outlier_ray_check.json').read_text()))
    summary=dict(settings=settings,receiver_count=len(rec),query_count=len(df),active_count=int((df.raw>.01).sum()),overall=error(df.raw.to_numpy(),df.packed.to_numpy()),bootstrap=bootstrap(df),groups=grouped,stages=stages,correction=correction,kernel_comparison=kernelbest,gamma_sensitivity=gamma_rows,
        runtime=dict(python=sys.version,platform=platform.platform(),logical_cpu=os.cpu_count(),numpy=np.__version__))
    for name,sub in [('synthetic',df[df.split=='validation']),('external',df[df.split=='external'])]:summary[name]=dict(**error(sub.raw.to_numpy(),sub.packed.to_numpy()),receivers=sub.id.nunique(),bootstrap=bootstrap(sub))
    c.old.atomic_json(out/'scientific_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['receiver_count','query_count','active_count','overall','synthetic','external','correction']},indent=2),flush=True)

def secondary():
    out=RESULTS/'analysis';out.mkdir(exist_ok=True)
    hourly=[]
    for p in sorted((RESULTS/'validation').glob('*/*/hourly.npz')):
        z=np.load(p);m=json.loads((p.parent/'metadata.json').read_text())
        beam=np.maximum(z['dni'],0)*np.maximum(z['incidence'],0)*(z['el']>0)
        truth=beam*(1-z['shade'][:,0]);pred=beam*(1-z['shade'][:,1]);delta=pred-truth
        hourly.append(dict(scene=m['scene'],terrain=m['terrain'],label=m['label'],kind='wall' if m['kind']=='wall' else 'roof',exact_beam_kwh_m2=float(truth.sum()/1000),estimated_beam_kwh_m2=float(pred.sum()/1000),beam_bias_kwh_m2=float(delta.sum()/1000),beam_relative_pct=float(100*delta.sum()/truth.sum()) if truth.sum()>1000 else np.nan,
            unshaded_beam_kwh_m2=float(beam.sum()/1000),annual_abs_hourly_error_kwh_m2=float(np.abs(delta).sum()/1000),solar_weighted_mae=float(np.abs(delta).sum()/beam.sum()) if beam.sum()>0 else np.nan,
            excess_shading_energy_kwh_m2=float(np.maximum(-delta,0).sum()/1000),missed_shading_energy_kwh_m2=float(np.maximum(delta,0).sum()/1000),peak_beam_error_w_m2=float(np.max(np.abs(delta))),front_daylight=int(z['front_daylight'].sum())))
    c.old.atomic_csv(out/'hourly_energy_metrics.csv',hourly)
    comparisons=[]
    for p in sorted((RESULTS/'validation').glob('*/*/comparison.json')):
        j=json.loads(p.read_text())
        for row in j['methods']:comparisons.append(dict(scene=j['scene'],label=j['label'],terrain=j['terrain'],split=j['split'],**row))
    c.old.atomic_csv(out/'method_comparison.csv',comparisons)
    thermal=[]
    for p in sorted((RESULTS/'thermal').glob('*/*/exact/summary.json')):
        other=p.parent.parent/'lobe/summary.json'
        if not other.exists():continue
        a=json.loads(p.read_text());b=json.loads(other.read_text());row=dict(scene=p.parents[2].name,label=p.parents[1].name)
        for key in ['heating_kwh','cooling_kwh','peak_heating_w','peak_cooling_w']:
            row['exact_'+key]=a[key];row['estimated_'+key]=b[key];row['delta_'+key]=b[key]-a[key]
            row['relative_pct_'+key]=100*(b[key]-a[key])/a[key] if a[key]>(1. if key.endswith('kwh') else 10.) else np.nan
        za=np.load(p.parent/'hourly_outputs.npz');zb=np.load(other.parent/'hourly_outputs.npz')
        row['hourly_heating_mae_w']=float(np.abs(zb['heating_w']-za['heating_w']).mean());row['hourly_cooling_mae_w']=float(np.abs(zb['cooling_w']-za['cooling_w']).mean());row['maximum_coincident_cooling_error_w']=float(np.abs(zb['cooling_w']-za['cooling_w']).max())
        row['exact_warnings']=a['warnings'];row['estimated_warnings']=b['warnings'];thermal.append(row)
    c.old.atomic_csv(out/'thermal_differences.csv',thermal)
    costs=[]
    comp=pd.DataFrame(comparisons)
    if len(comp):
        for (scene,label),sub in comp.groupby(['scene','label']):
            by={r['method']:r for r in sub.to_dict('records')};lobe=by['lobe_biweight']
            hm=[h for h in hourly if h['scene']==scene and h['label']==label];annual=hm[0]['front_daylight'] if hm else 4000
            for name,baseline in by.items():
                if name=='lobe_biweight':continue
                delta_t=baseline['query_s']-lobe['query_s'];delta_p=lobe['build_s']+lobe['load_s']-baseline['build_s']
                qstar=max(0,int(np.ceil(delta_p/delta_t))) if delta_t>0 else np.nan
                costs.append(dict(scene=scene,label=label,baseline=name,lobe_preprocessing_s=lobe['build_s']+lobe['load_s'],baseline_preprocessing_s=baseline['build_s'],crossover_queries=qstar,annual_queries=annual,
                    lobe_annual_s=lobe['build_s']+lobe['load_s']+annual*lobe['query_s'],baseline_annual_s=baseline['build_s']+annual*baseline['query_s'],query_speed_ratio=baseline['query_s']/lobe['query_s']))
    c.old.atomic_csv(out/'workflow_costs.csv',costs)
    c.old.atomic_json(out/'secondary_counts.json',dict(hourly=len(hourly),comparison_receivers=len(comparisons)//7,thermal_pairs=len(thermal)))
    print('SECONDARY',len(hourly),len(comparisons),len(thermal),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['primary','secondary','all']);args=p.parse_args()
    if args.phase in ['primary','all']:analysis()
    if args.phase in ['secondary','all']:secondary()
