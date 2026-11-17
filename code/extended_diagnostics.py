"""Explicitly post-hoc diagnostics; no change to the frozen primary method."""
import json,time,io
from pathlib import Path
import numpy as np
import pandas as pd
import revision_core as c
from run_revision import RESULTS,bundles,error,candidate_lists
from analyse_results import load_all

def run():
    out=RESULTS/'analysis';df,rec=load_all();rows=[];worst=[]
    for split,sub in [('synthetic',df[df.split=='validation']),('external',df[df.split=='external'])]:
        for stage,ref,estimate in [('final','raw','packed'),('representation','topk','float')]:
            conditions={'elevation <10°':sub.el<10,'elevation 10–30°':sub.el.between(10,30,inclusive='left'),
              'elevation 30–60°':sub.el.between(30,60,inclusive='left'),'elevation ≥60°':sub.el>=60,
              'incidence cosine <0.2':sub.incidence<.2,'incidence cosine ≥0.2':sub.incidence>=.2,
              'angular contrast >0.25':sub.transition_contrast>.25,'angular contrast ≤0.25':sub.transition_contrast<=.25,
              'one active retained caster':sub.multiplicity==1,'2–5 active retained casters':sub.multiplicity.between(2,5),'≥6 active retained casters':sub.multiplicity>=6,
              'wall':sub.kind=='wall','flat roof':(sub.kind=='roof')&(sub.pitch_deg<5),'pitched roof':(sub.kind=='roof')&(sub.pitch_deg>=5)}
            for condition,mask in conditions.items():
                ss=sub[mask];rows.append(dict(split=split,stage=stage,condition=condition,**error(ss[ref].to_numpy(),ss[estimate].to_numpy())))
        for stage,ref,estimate in [('proxy','raw','proxy'),('radius','proxy','radius'),('topk','radius','topk'),('representation','topk','float'),('quantisation','float','packed')]:
            if split=='external' and stage=='proxy':continue
            errors=(sub[estimate]-sub[ref]).abs();ii=errors.idxmax()
            worst.append(dict(sub.loc[ii].to_dict(),split=split,stage=stage,absolute_stage_error=float(errors.loc[ii]),reference=float(sub.loc[ii,ref]),estimate=float(sub.loc[ii,estimate])))
    c.old.atomic_csv(out/'stratified_error_conditions.csv',rows);c.old.atomic_json(out/'stage_worst_cases.json',dict(cases=worst))
    settings=json.loads((RESULTS/'calibration_final.json').read_text());wanted={(r['scene'],r['label']) for r in json.loads((RESULTS/'frozen_comparison_manifest.json').read_text())['receivers']}
    sweep=[];profiles=[];archives=[]
    for item,b in bundles():
        if b.split=='calibration':continue
        # Scene-level proxy construction is an additional, separately measured
        # cost; it is not silently charged to the raw-geometry baselines.
        proxy_s=0.
        if b.split!='external':
            t=time.perf_counter()
            proxy=c.old.SHADING_PROFILES.build_shading_profile(scene=b.scene,profile_mode='barrier_proxy',receiver_surface_kinds=('wall','flat_roof','pitched_roof'),decomposition_mode='merged',barrier_vertical_top_mode='eave',include_roof_casters=True,include_tree_casters=False,include_original_building_wall_casters=False,include_nonbuilding_casters=False,optimize_rotation=True,rotation_cost_mode='total_length',deduplicate_barriers=True,verbose=False)
            proxy_s=time.perf_counter()-t
        profiles.append(dict(scene=item['key'],split=b.split,receivers=len(b.receivers),additional_proxy_s=proxy_s,raw_building_casters=sum(p.object_kind=='building' for p in b.raw_profile.casters),proxy_building_casters=sum(p.object_kind=='building' for p in b.proxy_profile.casters)))
        archive={'params_i16':[],'caster_sids':[],'kind_codes':[],'receiver_meta':[],'receiver_id':[],'valid_count':[]}
        for label,r in b.receivers.items():
            directory=RESULTS/'validation'/item['key']/label;z=np.load(directory/'grid.npz');cache=c.old.receiver_cache(r);n=len(z['params']);k=settings['topk']
            padded=np.zeros((k,5),np.int16);padded[:n]=z['packed'];ids=np.zeros(k,np.uint32);ids[:n]=z['selected_sids'];kinds=np.zeros(k,np.uint8)
            sidmap={p.sid:p for p in b.proxy_profile.casters};kinds[:n]=[1 if sidmap[s].surface_kind=='barrier_wall_proxy' else 2 for s in z['selected_sids']]
            for key,val in dict(params_i16=padded,caster_sids=ids,kind_codes=kinds,receiver_meta=np.r_[r.center,cache.normal_azimuth_deg].astype(np.float32),receiver_id=np.uint32(r.sid),valid_count=np.uint16(n)).items():archive[key].append(val)
            if (item['key'],label) not in wanted:continue
            fits=np.load(directory/'fits.npz');rank=np.lexsort((fits['sids'],-fits['score'],-fits['mass'],-fits['peak']))
            for cap in [32,64,112,160,320,640,896]:
                chosen=rank[:cap];pars=fits[settings['kernel']][chosen];packed=c.old.quantize_params(pars,cache.normal_azimuth_deg);pq=c.old.dequantize_params(packed,float(np.float32(cache.normal_azimuth_deg)))
                pred=c.predict(pq,z['az'],z['el'],settings['gamma'],settings['kernel'])
                sweep.append(dict(scene=item['key'],split=b.split,terrain=b.terrain,label=label,K=cap,actual_lobes=len(chosen),fixed_record_bytes=22+15*cap,exploratory_after_primary_validation=True,**error(z['values'][:,0],pred)))
        archive={key:np.asarray(val) for key,val in archive.items()};dest=RESULTS/'archives'/(item['key']+'.npz');c.old.atomic_npz(dest,**archive)
        archives.append(dict(scene=item['key'],receivers=len(b.receivers),payload_bytes=sum(v.nbytes for v in archive.values()),compressed_bytes=dest.stat().st_size,compressed_bytes_per_receiver=dest.stat().st_size/len(b.receivers)))
    c.old.atomic_csv(out/'exploratory_K_sensitivity.csv',sweep);c.old.atomic_csv(out/'scene_preprocessing.csv',profiles);c.old.atomic_csv(out/'multi_receiver_archives.csv',archives)
    print('EXPLORATORY',len(sweep),'archive scenes',len(archives),flush=True)

if __name__=='__main__':run()
