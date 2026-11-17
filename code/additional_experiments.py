"""Frozen hourly subset, lookup/Embree comparisons, timing, and diagnostics."""
from __future__ import annotations
import json,pickle,time,argparse,io,platform,os,sys,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
import revision_core as c
from run_revision import bundles,error
ROOT=c.ROOT
RESULTS=ROOT/'final_results'

@njit(cache=True)
def lobe_scalar(params,az,el,gamma):
    # Final calibration selected biweight; assert kernel in calling workflow.
    logp=0.;largest=0.;total=0.
    for p in params:
        da=(az-p[0]+180)%360-180
        q=(da/max(.05,p[2]))**2+((el-p[1])/max(.05,p[3]))**2
        val=p[4]*(1-q)**2 if q<1 else 0.
        logp+=np.log1p(-min(1-1e-12,max(0.,val)))
        largest=max(largest,val);total+=val
    return max(largest,min(1.,total,-np.expm1(gamma*logp)))

@njit(cache=True)
def lut_scalar(grid,az,el,step):
    x=(az%360)/step;y=min(max(el,0.),90.)/step
    i=int(x);j=min(int(y),grid.shape[0]-2);dx=x-i;dy=y-j
    ni=(i+1)%grid.shape[1]
    return ((1-dx)*(1-dy)*grid[j,i]+dx*(1-dy)*grid[j,ni]+(1-dx)*dy*grid[j+1,i]+dx*dy*grid[j+1,ni])/65535.

def freeze_subsets():
    hourly=[];comparators=[]
    for item,b in bundles():
        if b.split=='calibration':continue
        nh=8 if b.split=='external' else 2
        nc=4 if b.split=='external' else 1
        for kind in ('wall','roof'):
            for i in range(nh):hourly.append(dict(scene=item['key'],label=f'{kind}_{i:02d}'))
            for i in range(nc):comparators.append(dict(scene=item['key'],label=f'{kind}_{i:02d}'))
    for name,receivers in [('hourly',hourly),('comparison',comparators)]:
        path=RESULTS/f'frozen_{name}_manifest.json'
        if not path.exists():c.old.atomic_json(path,dict(created=c.old.now_iso(),rule='first labelled receivers in geometry-stratified fixed sample; no outcome filtering',receivers=receivers))

def weather():
    import pvlib
    w,meta=pvlib.iotools.read_epw(ROOT/'data/rotterdam_pvgis_tmy.epw',coerce_year=2021)
    midpoint=w.index+pd.Timedelta(minutes=30)
    solar=pvlib.solarposition.get_solarposition(midpoint,meta['latitude'],meta['longitude'],meta['altitude'],temperature=w.temp_air.to_numpy())
    return w,meta,solar.azimuth.to_numpy(),solar.elevation.to_numpy()

def hourly():
    settings=json.loads((RESULTS/'calibration_final.json').read_text());freeze_subsets()
    wanted={(r['scene'],r['label']) for r in json.loads((RESULTS/'frozen_hourly_manifest.json').read_text())['receivers']}
    w,meta,az,el=weather();dd=c.directions(az,el)
    c.old.atomic_json(RESULTS/'weather_provenance.json',dict(source='European Commission JRC PVGIS v5.3 TMY API; EPW',url='https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat=51.9225&lon=4.4792&outputformat=epw',metadata=meta,solar_time='hour midpoint; local standard time UTC+1; year 2021; pvlib solarposition.get_solarposition',hours=len(w)))
    for item,b in bundles():
        for label,r in b.receivers.items():
            if (item['key'],label) not in wanted:continue
            directory=RESULTS/'validation'/item['key']/label;out=directory/'hourly.npz'
            if out.exists():continue
            if not (directory/'grid.npz').exists():continue
            start=time.perf_counter();grid=np.load(directory/'grid.npz');pq=c.old.dequantize_params(grid['packed'],float(np.float32(c.old.receiver_cache(r).normal_azimuth_deg)))
            g=c.prepare(r,c.old.building_casters(b.raw_profile,r));inc=dd@r.normal
            # Include grazing sun in energy propagation; only physically front-facing.
            mask=(el>0)&(inc>1e-6)
            shade=np.zeros((8760,2));times=[]
            for j in np.flatnonzero(mask):
                t=time.perf_counter();shade[j,0]=c.exact(g,az[j],el[j]);times.append(time.perf_counter()-t)
            shade[mask,1]=c.predict(pq,az[mask],el[mask],settings['gamma'],settings['kernel'])
            c.old.atomic_npz(out,az=az,el=el,incidence=inc,dni=w.dni.to_numpy(),dhi=w.dhi.to_numpy(),temperature=w.temp_air.to_numpy(),shade=shade,front_daylight=mask,exact_query_times=np.asarray(times))
            c.old.atomic_json(directory/'hourly_meta.json',dict(elapsed_s=time.perf_counter()-start,front_daylight_queries=int(mask.sum()),all_hours=8760,**error(shade[mask,0],shade[mask,1])))
            print('HOURLY',item['key'],label,round(time.perf_counter()-start,1),flush=True)

def time_queries(fn,az,el,repeats=7):
    # Median repetition of a fixed mixed-direction workload. No result cache.
    for a,e in zip(az[:5],el[:5]):fn(a,e)
    measurements=[];checksum=0.
    for rep in range(repeats):
        start=time.perf_counter()
        for a,e in zip(az,el):checksum+=fn(a,e)
        measurements.append((time.perf_counter()-start)/len(az))
    return dict(median_s=float(np.median(measurements)),min_s=float(np.min(measurements)),max_s=float(np.max(measurements)),checksum=checksum)

def compare():
    freeze_subsets();settings=json.loads((RESULTS/'calibration_final.json').read_text());assert settings['kernel']=='biweight'
    wanted={(r['scene'],r['label']) for r in json.loads((RESULTS/'frozen_comparison_manifest.json').read_text())['receivers']}
    for item,b in bundles():
        for label,r in b.receivers.items():
            if (item['key'],label) not in wanted:continue
            directory=RESULTS/'validation'/item['key']/label;out=directory/'comparison.json'
            if out.exists():continue
            if not (directory/'grid.npz').exists():continue
            start=time.perf_counter();z=np.load(directory/'grid.npz');a,e=z['az'],z['el'];truth=z['values'][:,0]
            meta=json.loads((directory/'metadata.json').read_text());cache=c.old.receiver_cache(r)
            pars=c.old.dequantize_params(z['packed'],float(np.float32(cache.normal_azimuth_deg)));gamma=settings['gamma']
            # JIT compilation is not included in warmed query timings; recorded separately.
            t=time.perf_counter();lobe_scalar(pars,float(a[0]),float(e[0]),gamma);jit_s=time.perf_counter()-t
            checked=np.array([lobe_scalar(pars,float(aa),float(ee),gamma) for aa,ee in zip(a,e)])
            if not np.allclose(checked,z['values'][:,6],atol=1e-12,rtol=1e-12):raise ValueError('compiled/packed prediction disagreement')
            tq=time_queries(lambda aa,ee:lobe_scalar(pars,aa,ee,gamma),a,e)
            tn=time_queries(lambda aa,ee:c.bounded_lobe_scalar(settings['kernel'],pars,aa,ee,gamma),a,e)
            raw=c.old.building_casters(b.raw_profile,r);g=c.prepare(r,raw)
            tx=time_queries(lambda aa,ee:c.exact(g,aa,ee),a,e,3)
            # Preserve the submitted fixed-width record with the recalibrated K.
            packed=np.zeros((settings['topk'],5),np.int16);packed[:len(pars)]=z['packed']
            ids=np.zeros(settings['topk'],np.uint32);ids[:len(pars)]=z['selected_sids']
            kinds=np.zeros(settings['topk'],np.uint8)
            by_sid={p.sid:p for p in b.proxy_profile.casters}
            kinds[:len(pars)]=[1 if by_sid[sid].surface_kind=='barrier_wall_proxy' else 2 for sid in z['selected_sids']]
            receiver_meta=np.r_[r.center,cache.normal_azimuth_deg].astype(np.float32)
            archive=dict(parameters=packed,surface_id=ids,kind=kinds,receiver_meta=receiver_meta,receiver_id=np.array([r.sid],np.uint32),valid_count=np.array([len(pars)],np.uint16))
            packed_bytes=sum(v.nbytes for v in archive.values())
            buf=io.BytesIO();np.savez_compressed(buf,**archive)
            npz_bytes=len(buf.getvalue());blob=buf.getvalue()
            loadtimes=[]
            for rep in range(30):
                tt=time.perf_counter()
                with np.load(io.BytesIO(blob)) as zz:c.old.dequantize_params(zz['parameters'][:int(zz['valid_count'][0])],float(zz['receiver_meta'][3]))
                loadtimes.append(time.perf_counter()-tt)
            # Cold build includes candidate retrieval, sampling, every candidate
            # fit, ranking and checkpoint I/O; source scene loading is shared.
            from run_revision import candidate_lists
            tb=time.perf_counter();_,_,cold_casters=candidate_lists(b,r,settings['radius_m'])
            # A fresh path prevents an interrupted comparison from accidentally
            # measuring a cache load as a cold fit on its next invocation.
            cold=c.fit_receiver(r,cold_casters,directory/('timing_cold_'+uuid.uuid4().hex+'.npz'),(settings['kernel'],))
            cold['rank'][:settings['topk']]
            c.old.quantize_params(cold['params'][settings['kernel']][cold['rank'][:settings['topk']]],cache.normal_azimuth_deg)
            cold_s=time.perf_counter()-tb
            if not np.allclose(cold['params'][settings['kernel']][cold['rank'][:settings['topk']]],z['params']):raise ValueError('non-reproducible cold fit')
            results=[dict(method='lobe_biweight',**error(truth,z['values'][:,6]),query_s=tq['median_s'],numpy_query_s=tn['median_s'],build_s=cold_s,fit_sum_s=meta['fitting_s'],packed_bytes=packed_bytes,npz_bytes=npz_bytes,load_s=float(np.median(loadtimes)),jit_first_call_s=jit_s)]
            predictions={'truth':truth,'lobe':z['values'][:,6]}
            for step in [10.,5.]:
                path=directory/f'lookup_{int(step)}.npz'
                if path.exists():
                    zz=np.load(path);grid=zz['grid'];build=float(zz['build_s'][0])
                else:
                    startbuild=time.perf_counter();aa,ee=np.meshgrid(np.arange(0,360,step),np.arange(0,90+.1,step));vals=np.array([c.exact(g,aa0,ee0) if r.normal@d>1e-6 else 0 for aa0,ee0,d in zip(aa.ravel(),ee.ravel(),c.directions(aa.ravel(),ee.ravel()))]);grid=np.rint(vals.reshape(aa.shape)*65535).astype(np.uint16);build=time.perf_counter()-startbuild
                    c.old.atomic_npz(path,grid=grid,build_s=np.array([build]))
                pred=np.array([lut_scalar(grid,float(aa),float(ee),step) for aa,ee in zip(a,e)])
                t=time_queries(lambda aa,ee:lut_scalar(grid,aa,ee,step),a,e)
                results.append(dict(method=f'angular_lookup_{int(step)}deg',**error(truth,pred),query_s=t['median_s'],build_s=build+g.preparation_s,packed_bytes=grid.nbytes+8,npz_bytes=path.stat().st_size))
                predictions[f'lookup_{int(step)}']=pred
            engine,bvh_s,mesh=c.ray_engine(raw)
            for n in [256,1024,4096]:
                rt=[];pred=None
                for rep in range(3):
                    tt=time.perf_counter();pred=c.ray_fractions(engine,r,a,e,n);rt.append((time.perf_counter()-tt)/len(a))
                results.append(dict(method=f'Embree_{n}_samples',**error(truth,pred),query_s=float(np.median(rt)),build_s=bvh_s,geometry_array_bytes=int(mesh.vertices.nbytes+mesh.faces.nbytes),bvh_memory_not_measured=True))
                predictions[f'embree_{n}']=pred
            results.append(dict(method='compiled_polygon_union',**error(truth,truth),query_s=tx['median_s'],build_s=g.preparation_s,geometry_array_bytes=int(g.vertices.nbytes+g.counts.nbytes+g.sids.nbytes+g.distances.nbytes)))
            c.old.atomic_npz(directory/'comparison_predictions.npz',az=a,el=e,**predictions)
            c.old.atomic_json(out,dict(scene=item['key'],label=label,terrain=b.terrain,split=b.split,receiver=r.sid,triangles=len(mesh.faces),methods=results,elapsed_s=time.perf_counter()-start,timing_repeats=7,ray_repeats=3,ray_backend='trimesh.ray.ray_pyembree.RayMeshIntersector / embreex 4.4.0; CPU',own_building_excluded=True))
            print('COMPARED',item['key'],label,round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','hourly','compare']);args=p.parse_args()
    {'freeze':freeze_subsets,'hourly':hourly,'compare':compare}[args.phase]()
