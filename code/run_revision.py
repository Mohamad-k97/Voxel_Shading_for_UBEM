"""Restartable corrected calibration and untouched final receiver evaluation."""
from __future__ import annotations
import json,pickle,time,argparse,platform,os
from pathlib import Path
import numpy as np
import pandas as pd
import revision_core as c
ROOT=c.ROOT
RESULTS=ROOT/'final_results'
R_LEVELS=np.array([50,100,150,200,250,300,350,500],dtype=float)
K_LEVELS=np.array([16,32,48,64,80,112,160,224,320,448,640,896],dtype=int)
GAMMAS=np.arange(.3,1.201,.1)

def bundles(split=None):
    manifest=json.loads((RESULTS/'frozen_sample_manifest.json').read_text())['scenes']
    for item in manifest:
        if split and item['split']!=split:continue
        with (RESULTS/'scenes'/f"{item['key']}.pkl").open('rb') as f:b=pickle.load(f)
        yield item,b

def angle_grid(receiver,validation=True):
    if validation: azs,els=np.arange(2.5,360,10.),np.arange(2.5,90,5.)
    else: azs,els=np.arange(0,360,15.),np.array([5,15,25,35,45,60,75,85.])
    a,e=np.meshgrid(azs,els); a,e=a.ravel(),e.ravel()
    valid=c.directions(a,e)@receiver.normal>.02
    return a[valid],e[valid]

def candidate_lists(b,r,radius):
    raw=c.old.building_casters(b.raw_profile,r)
    proxy=c.old.building_casters(b.proxy_profile,r)
    radius_c=[p for p in proxy if c.old.horizontal_distance(r,p)<=radius]
    return raw,proxy,radius_c

def error(reference,estimate):
    delta=np.asarray(estimate)-np.asarray(reference); active=np.asarray(reference)>.01
    out={}
    for label,mask in [('all',np.ones(len(delta),dtype=bool)),('active',active)]:
        d=delta[mask]
        for key,val in [('n',len(d)),('mae',np.mean(np.abs(d)) if len(d) else np.nan),('bias',np.mean(d) if len(d) else np.nan),('rmse',np.sqrt(np.mean(d*d)) if len(d) else np.nan),('p95',np.percentile(np.abs(d),95) if len(d) else np.nan),('max',np.max(np.abs(d)) if len(d) else np.nan)]:out[label+'_'+key]=val
    out['excess_shading_mean']=np.maximum(delta,0).mean();out['missed_shading_mean']=np.maximum(-delta,0).mean()
    out['false_shade_005_rate']=np.mean((reference<=.01)&(estimate>.05))
    out['missed_shade_005_rate']=np.mean((reference>.05)&(estimate<=.01))
    return out

def radius_calibration():
    rows=[]
    for item,b in bundles('calibration'):
        for label,r in b.receivers.items():
            path=RESULTS/'calibration'/item['key']/f'{label}_radius.npz'
            if path.exists():z=dict(np.load(path))
            else:
                proxy=c.old.building_casters(b.proxy_profile,r);g=c.prepare(r,proxy);a,e=angle_grid(r,False)
                vals=np.zeros((len(a),len(R_LEVELS)+1))
                for i,d in enumerate(c.directions(a,e)):
                    polys,idx,_=c.exact_parts(g,d);vals[i,0]=c.fraction(polys,g.cache.area)
                    for j,rad in enumerate(R_LEVELS):vals[i,j+1]=c.fraction([p for p,ii in zip(polys,idx) if g.distances[ii]<=rad],g.cache.area)
                z=dict(az=a,el=e,values=vals);c.old.atomic_npz(path,**z)
            for j,rad in enumerate(R_LEVELS):rows.append(dict(scene=item['key'],label=label,terrain=b.terrain,radius=rad,**error(z['values'][:,0],z['values'][:,j+1])))
    df=pd.DataFrame(rows);c.old.atomic_csv(RESULTS/'radius_calibration.csv',rows)
    agg=df.groupby('radius')[['active_mae','active_p95']].mean()
    ok=agg.index[(agg.active_mae<=.005)&(agg.active_p95<=.02)]
    knee=float(ok[0]) if len(ok) else float(R_LEVELS[-1]);chosen=float(R_LEVELS[min(np.where(R_LEVELS==knee)[0][0]+1,len(R_LEVELS)-1)])
    return knee,chosen

def calibration():
    out=RESULTS/'calibration_final.json'
    if out.exists():print(out.read_text(),flush=True);return
    knee,radius=radius_calibration();print('RADIUS',knee,radius,flush=True)
    records=[];krows=[];allpars=[]
    for item,b in bundles('calibration'):
        for label,r in b.receivers.items():
            directory=RESULTS/'calibration'/item['key'];directory.mkdir(parents=True,exist_ok=True)
            raw,proxy,cs=candidate_lists(b,r,radius)
            fit=c.fit_receiver(r,cs,directory/f'{label}_fits.npz',c.old.KERNELS)
            path=directory/f'{label}_exact.npz'
            if path.exists():z=dict(np.load(path))
            else:
                a,e=angle_grid(r,False);g=c.prepare(r,cs);rankpos=np.empty(len(cs),int);rankpos[fit['rank']]=np.arange(len(cs))
                truth=np.zeros(len(a));top=np.zeros((len(a),len(K_LEVELS)))
                for i,d in enumerate(c.directions(a,e)):
                    polys,idx,_=c.exact_parts(g,d);truth[i]=c.fraction(polys,g.cache.area)
                    for j,kk in enumerate(K_LEVELS):top[i,j]=c.fraction([p for p,ii in zip(polys,idx) if rankpos[ii]<kk],g.cache.area)
                z=dict(az=a,el=e,truth=truth,topk=top);c.old.atomic_npz(path,**z)
            for j,kk in enumerate(K_LEVELS):krows.append(dict(scene=item['key'],terrain=b.terrain,label=label,K=int(kk),**error(z['truth'],z['topk'][:,j])))
            allpars.append((item,b.terrain,label,r,fit,z))
            print('CAL_FIT',item['key'],label,len(cs),round(fit['elapsed'].sum(),2),flush=True)
    kd=pd.DataFrame(krows);agg=kd.groupby('K')[['active_mae','active_p95']].mean();ok=agg.index[(agg.active_mae<=.005)&(agg.active_p95<=.02)]
    kknee=int(ok[0]) if len(ok) else int(K_LEVELS[-1]);topk=int(K_LEVELS[min(np.where(K_LEVELS==kknee)[0][0]+1,len(K_LEVELS)-1)])
    c.old.atomic_csv(RESULTS/'topk_calibration.csv',krows)
    for item,terrain,label,r,fit,z in allpars:
        rank=fit['rank'][:topk]
        for kernel in c.old.KERNELS:
            p=fit['params'][kernel][rank]
            for gamma in GAMMAS:
                pred=c.predict(p,z['az'],z['el'],gamma,kernel)
                records.append(dict(scene=item['key'],terrain=terrain,label=label,kind=label.split('_')[0],kernel=kernel,gamma=round(float(gamma),2),**error(z['truth'],pred)))
    c.old.atomic_csv(RESULTS/'kernel_gamma_calibration.csv',records)
    df=pd.DataFrame(records);agg=df.groupby(['kernel','gamma'])['active_mae'].mean().sort_values();kernel,gamma=agg.index[0]
    result=dict(radius_knee_m=knee,radius_m=radius,topk_knee=kknee,topk=topk,kernel=kernel,gamma=float(gamma),aggregation_version='frechet_bounds_v2',selection_objective='equal-receiver mean active MAE; four calibration morphologies, four walls and four roofs each',created=c.old.now_iso(),final_data_not_used=True)
    c.old.atomic_json(out,result);print(json.dumps(result),flush=True)

def evaluate_receiver(item,b,label,r,settings):
    directory=RESULTS/'validation'/item['key']/label;directory.mkdir(parents=True,exist_ok=True)
    output=directory/'grid.npz'
    if output.exists():return
    start=time.perf_counter()
    raw,proxy,cs=candidate_lists(b,r,settings['radius_m'])
    fit=c.fit_receiver(r,cs,directory/'fits.npz',(settings['kernel'],))
    rank=fit['rank'][:settings['topk']];selected={p.sid for p in np.array(cs,dtype=object)[rank]}
    pars=fit['params'][settings['kernel']][rank];cache=c.old.receiver_cache(r)
    quant=c.old.quantize_params(pars,cache.normal_azimuth_deg);pq=c.old.dequantize_params(quant,float(np.float32(cache.normal_azimuth_deg)))
    a,e=angle_grid(r,True);dd=c.directions(a,e)
    rawg=c.prepare(r,raw);proxyg=rawg if b.split=='external' else c.prepare(r,proxy)
    values=np.zeros((len(a),7));timings=np.zeros((len(a),2));single_mae=np.zeros(len(a));single_max=np.zeros(len(a));overlap=np.zeros(len(a));multiplicity=np.zeros(len(a),int)
    sid_to_rank={sid:j for j,sid in enumerate(fit['sids'][rank])}
    for i,d in enumerate(dd):
        t=time.perf_counter();polys,idx,fracs=c.exact_parts(rawg,d);values[i,0]=c.fraction(polys,rawg.cache.area);timings[i,0]=time.perf_counter()-t
        t=time.perf_counter()
        if b.split!='external':polys,idx,fracs=c.exact_parts(proxyg,d)
        values[i,1]=c.fraction(polys,proxyg.cache.area)
        values[i,2]=c.fraction([p for p,ii in zip(polys,idx) if proxyg.distances[ii]<=settings['radius_m']],proxyg.cache.area)
        top_parts=[p for p,ii in zip(polys,idx) if proxyg.sids[ii] in selected]
        values[i,3]=c.fraction(top_parts,proxyg.cache.area)
        singles=np.zeros(len(rank))
        for ii,f in zip(idx,fracs):
            if proxyg.sids[ii] in sid_to_rank:singles[sid_to_rank[proxyg.sids[ii]]]=f
        values[i,4]=c.bounded_union(singles,settings['gamma'])
        approx=c.old.kernel_values_many(settings['kernel'],a[i],e[i],pars)
        single_mae[i]=np.mean(np.abs(approx-singles)) if len(singles) else 0
        single_max[i]=np.max(np.abs(approx-singles)) if len(singles) else 0
        overlap[i]=singles.sum()-values[i,3];multiplicity[i]=np.sum(singles>.01)
        timings[i,1]=time.perf_counter()-t
    values[:,5]=c.predict(pars,a,e,settings['gamma'],settings['kernel'])
    values[:,6]=c.predict(pq,a,e,settings['gamma'],settings['kernel'])
    c.old.atomic_npz(output,az=a,el=e,values=values,timings=timings,single_mae=single_mae,single_max=single_max,overlap=overlap,multiplicity=multiplicity,
        params=pars,packed=quant,selected_sids=fit['sids'][rank])
    meta=dict(scene=item['key'],terrain=b.terrain,split=b.split,label=label,sid=r.sid,building=r.object_id,kind=r.surface_kind,area_m2=r.area,normal=r.normal,center=r.center,
        raw_casters=len(raw),proxy_casters=len(proxy),candidate_casters=len(cs),retained_casters=len(rank),positive_lobes=int(np.sum(fit['peak'][rank]>1e-8)),
        fitting_s=float(fit['elapsed'].sum()),prepare_raw_s=rawg.preparation_s,prepare_proxy_s=proxyg.preparation_s,elapsed_s=time.perf_counter()-start,
        raw_median_query_s=float(np.median(timings[:,0])),columns=['raw','proxy','radius','topk','exact_marginal_heuristic_union','lobe_float','lobe_quantized'])
    c.old.atomic_json(directory/'metadata.json',meta)
    print('VALIDATED',item['key'],label,'candidates',len(cs),'seconds',round(time.perf_counter()-start,1),flush=True)

def validation():
    settings=json.loads((RESULTS/'calibration_final.json').read_text())
    for item,b in bundles():
        if b.split=='calibration':continue
        for label,r in b.receivers.items():evaluate_receiver(item,b,label,r,settings)

def original_audit():
    path=RESULTS/'original_reference_audit.csv'
    if path.exists():return
    rows=[]
    for b in c.synthetic_bundles():
        if b.split!='validation':continue
        for label,r in b.receivers.items():
            raw=c.old.building_casters(b.raw_profile,r);proxy=c.old.building_casters(b.proxy_profile,r)
            rawg=c.prepare(r,raw);pg=c.prepare(r,proxy)
            oldrows=pd.read_csv(c.PACKAGE/'study_output/scenes'/c.old.scene_key(b.split,b.terrain,b.seed)/'receivers'/label/'validation_rows.csv')
            for _,row in oldrows.iterrows():
                az=float(row['azimuth_deg']);el=float(row['elevation_deg'])
                rows.append(dict(terrain=b.terrain,seed=b.seed,label=label,sid=r.sid,az=az,el=el,
                    old_raw=float(row['raw_full']),old_proxy=float(row['proxy_full']),corrected_raw=c.exact(rawg,az,el),corrected_proxy=c.exact(pg,az,el)))
    c.old.atomic_csv(path,rows)
    # Independent dense sample on the originally reported qualitative failure.
    b=[b for b in c.synthetic_bundles() if b.seed==307][0];r=b.receivers['roof']
    engine,_,_=c.ray_engine(c.old.building_casters(b.raw_profile,r))
    val=c.ray_fractions(engine,r,[180],[20],65536)[0]
    c.old.atomic_json(RESULTS/'original_outlier_ray_check.json',dict(raw_corrected=c.exact(c.prepare(r,c.old.building_casters(b.raw_profile,r)),180,20),ray_fraction_65536=float(val),old_raw=1.,old_proxy=0.))
    print('ORIGINAL_AUDIT',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['audit','calibrate','validate','all']);args=parser.parse_args()
    if args.phase in ('audit','all'):original_audit()
    if args.phase in ('calibrate','all'):calibration()
    if args.phase in ('validate','all'):validation()
