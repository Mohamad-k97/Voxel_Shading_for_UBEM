"""Single source for manuscript tables and quoted numerical summaries."""
import json,io,time
import numpy as np
import pandas as pd
import revision_core as c
from run_revision import RESULTS,error

def main():
    root=RESULTS/'analysis';summary=json.loads((root/'scientific_summary.json').read_text());comp=pd.read_csv(root/'method_comparison.csv')
    keys={'lobe_biweight':'lobe','angular_lookup_10deg':'lookup_10','angular_lookup_5deg':'lookup_5','Embree_256_samples':'embree_256','Embree_1024_samples':'embree_1024','Embree_4096_samples':'embree_4096','compiled_polygon_union':'truth'}
    observations=[]
    for path in (RESULTS/'validation').glob('*/*/comparison_predictions.npz'):
        z=np.load(path);meta=json.loads((path.parent/'metadata.json').read_text())
        observations.append((meta['split'],dict(z)))
    methods=[]
    for split in ['all','validation','external']:
        obs=[o for s,o in observations if split=='all' or s==split];truth=np.concatenate([o['truth'] for o in obs])
        for method,key in keys.items():
            pred=np.concatenate([o[key] for o in obs]);ss=comp[comp.method==method]
            if split!='all':ss=ss[ss.split==split]
            methods.append(dict(split=split,method=method,receivers=len(obs),median_query_us=ss.query_s.median()*1e6,median_build_s=ss.build_s.median(),**error(truth,pred)))
    c.old.atomic_csv(root/'pooled_method_summary.csv',methods)
    profiles=pd.read_csv(root/'scene_preprocessing.csv').set_index('scene');workflow=pd.read_csv(root/'workflow_costs.csv');archive_encoding={}
    for scene,ss in comp[comp.method=='lobe_biweight'].groupby('scene'):
        with np.load(RESULTS/'archives'/(scene+'.npz')) as z:
            # One-row encoding is charged conservatively to each receiver.
            archive={k:z[k][:1] for k in z.files}
        times=[]
        for _ in range(7):
            t=time.perf_counter();buf=io.BytesIO();np.savez_compressed(buf,**archive);times.append(time.perf_counter()-t)
        archive_encoding[scene]=float(np.median(times))
    costs=[]
    for row in workflow.to_dict('records'):
        common_share=profiles.loc[row['scene'],'additional_proxy_s']/profiles.loc[row['scene'],'receivers'];encoding=archive_encoding[row['scene']]
        row['proxy_scene_share_s']=common_share;row['archive_encoding_s']=encoding
        row['lobe_preprocessing_s']+=common_share+encoding;row['lobe_annual_s']+=common_share+encoding
        cc=comp[(comp.scene==row['scene'])&(comp.label==row['label'])].set_index('method');dt=cc.loc[row['baseline'],'query_s']-cc.loc['lobe_biweight','query_s'];dp=row['lobe_preprocessing_s']-row['baseline_preprocessing_s']
        row['crossover_queries']=max(0,int(np.ceil(dp/dt))) if dt>0 else np.nan
        row['baseline_over_lobe_annual_cost']=row['baseline_annual_s']/row['lobe_annual_s'];costs.append(row)
    costs=pd.DataFrame(costs);c.old.atomic_csv(root/'complete_workflow_costs.csv',costs.to_dict('records'))
    cost_summary=[]
    for method,ss in costs.groupby('baseline'):
        q=ss.crossover_queries.dropna()
        cost_summary.append(dict(method=method,median_crossover=float(q.median()) if len(q) else None,min_crossover=float(q.min()) if len(q) else None,max_crossover=float(q.max()) if len(q) else None,lobe_annual_wins=int((ss.lobe_annual_s<ss.baseline_annual_s).sum()),receivers=len(ss),median_annual_cost_ratio=float(ss.baseline_over_lobe_annual_cost.median()),median_query_speed_ratio=float(ss.query_speed_ratio.median())))
    archives=pd.read_csv(root/'multi_receiver_archives.csv');synthetic=profiles[profiles.split=='validation'];h=pd.read_csv(root/'hourly_energy_metrics.csv');thermal=pd.read_csv(root/'thermal_differences.csv')
    k=pd.read_csv(root/'exploratory_K_sensitivity.csv');ks=[]
    for (split,cap),ss in k.groupby(['split','K']):ks.append(dict(split=split,K=int(cap),active_mae=float(np.nansum(ss.active_mae*ss.active_n)/ss.active_n.sum()),bytes=22+15*int(cap)))
    header=dict(pooled_comparators=methods,costs=cost_summary,storage=dict(payload_bytes=2422,compressed_total_bytes=int(archives.compressed_bytes.sum()),compressed_bytes_per_receiver=float(archives.compressed_bytes.sum()/archives.receivers.sum()),compressed_scene_mean_min=float(archives.compressed_bytes_per_receiver.min()),compressed_scene_mean_max=float(archives.compressed_bytes_per_receiver.max())),
      synthetic_proxy_reduction_equal_scene_mean_pct=float(100*np.mean(1-synthetic.proxy_building_casters/synthetic.raw_building_casters)),
      candidate_min=int(pd.read_csv(root/'receiver_metrics.csv').candidate_casters.min()),candidate_max=int(pd.read_csv(root/'receiver_metrics.csv').candidate_casters.max()),
      numpy_query_median_us=float(comp[comp.method=='lobe_biweight'].numpy_query_s.median()*1e6),load_median_ms=float(comp[comp.method=='lobe_biweight'].load_s.median()*1e3),jit_first_call_max_s=float(comp[comp.method=='lobe_biweight'].jit_first_call_s.max()),
      hourly_count=len(h),thermal_count=len(thermal),cooling_percent_denominator_count=int(thermal.relative_pct_cooling_kwh.notna().sum()),peak_cooling_percent_denominator_count=int(thermal.relative_pct_peak_cooling_w.notna().sum()),K_sensitivity=ks)
    c.old.atomic_json(root/'report_values.json',header);print(json.dumps(header,indent=2),flush=True)

if __name__=='__main__':main()
