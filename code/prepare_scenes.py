"""Geometry-only preparation and frozen, outcome-independent receiver selection."""
from __future__ import annotations
import json, time, pickle, hashlib
from pathlib import Path
from collections import Counter, deque
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon
from shapely import constrained_delaunay_triangles
import revision_core as core

ROOT=core.ROOT

def convex_pieces(poly):
    """Constrained triangulation plus adjacent convex merging (no area loss)."""
    if not poly.interiors and abs(poly.convex_hull.area-poly.area)<max(1e-8,poly.area*1e-10):
        return [np.asarray(poly.simplify(1e-9,preserve_topology=True).exterior.coords)[:-1]]
    pieces={i:p for i,p in enumerate(constrained_delaunay_triangles(poly).geoms)}
    edges={}; pairs=deque(); next_id=len(pieces)
    def add(idx,p):
        coords=list(p.exterior.coords)[:-1]
        for a,b in zip(coords,coords[1:]+coords[:1]):
            key=tuple(sorted((tuple(np.round(a,8)),tuple(np.round(b,8)))))
            for other in edges.get(key,[]):
                if other in pieces:pairs.append((idx,other))
            edges.setdefault(key,[]).append(idx)
    for i,p in pieces.items():add(i,p)
    while pairs:
        a,b=pairs.popleft()
        if a not in pieces or b not in pieces:continue
        merged=pieces[a].union(pieces[b])
        if merged.geom_type!='Polygon' or merged.interiors:continue
        if abs(merged.convex_hull.area-merged.area)>max(1e-8,merged.area*1e-10):continue
        merged=merged.simplify(1e-9,preserve_topology=True)
        if len(merged.exterior.coords)>33:continue
        del pieces[a];del pieces[b]
        pieces[next_id]=merged;add(next_id,merged);next_id+=1
    return [np.asarray(p.exterior.coords)[:-1] for p in pieces.values()]

def cityjson_surfaces(path,name,lod):
    started=time.perf_counter()
    data=json.loads(Path(path).read_text())
    transform=data.get('transform',{})
    vertices=np.asarray(data['vertices'],dtype=float)*transform.get('scale',[1,1,1])+transform.get('translate',[0,0,0])
    # Translate to a local origin before numerical geometry (record transform).
    origin=np.r_[np.min(vertices[:,:2],axis=0),0.]
    vertices-=origin
    objs=data['CityObjects']; parents={}
    for key,obj in objs.items():
        if obj.get('type')=='Building':
            parents[key]=key
            for child in obj.get('children',[]): parents[child]=key
    building_ids={key:i+1 for i,key in enumerate(sorted(set(parents.values())))}
    faces=[]; log=Counter(); deviations=[]; rejected=[]; source_area=0.
    for object_index,(key,obj) in enumerate(sorted(objs.items())):
        if object_index%200==0:print(name,'objects',object_index,'faces',len(faces),flush=True)
        for geom in obj.get('geometry',[]):
            if str(geom.get('lod'))!=str(lod):continue
            semantics=geom.get('semantics',{})
            types=semantics.get('surfaces',[])
            if geom['type']=='Solid':
                surfaces=[(s,sv) for shell,svals in zip(geom['boundaries'],semantics.get('values',[])) for s,sv in zip(shell,svals)]
            elif geom['type']=='MultiSurface': surfaces=list(zip(geom['boundaries'],semantics.get('values',[])))
            else:
                log['unsupported_geometry']+=1;continue
            bid=building_ids[parents.get(key,key)]
            for surf_idx,(rings,sv) in enumerate(surfaces):
                log['source_surfaces']+=1
                semantic=types[sv]['type'] if sv is not None else 'Unknown'
                if semantic not in ('WallSurface','RoofSurface'):
                    log['ground_or_other_omitted']+=1;continue
                cleaned=[]
                for ring in rings:
                    v=vertices[ring]
                    if len(v)>1 and np.linalg.norm(v[0]-v[-1])<1e-8:v=v[:-1]
                    if len(v)>1:
                        mask=np.r_[True,np.linalg.norm(np.diff(v,axis=0),axis=1)>1e-8]
                        if not mask.all(): log['rings_with_duplicate_vertices']+=1
                        v=v[mask]
                    cleaned.append(v)
                vv=cleaned[0]
                if len(vv)<3:
                    log['degenerate_omitted']+=1;continue
                center=vv.mean(axis=0)
                _,singular,vh=np.linalg.svd(vv-center,full_matrices=False)
                if len(singular)<2 or singular[1]<1e-8:
                    log['degenerate_omitted']+=1;continue
                normal=vh[-1]
                winding=np.sum(np.cross(vv-center,np.roll(vv-center,-1,axis=0)),axis=0)
                if normal@winding<0:normal=-normal
                if semantic=='RoofSurface' and normal[2]<0:normal=-normal;log['roof_normal_reoriented']+=1
                u,v=core.old.PROFILER.make_plane_basis(normal)
                planarity=max(np.max(np.abs((ring-center)@normal)) for ring in cleaned if len(ring))
                deviations.append(float(planarity))
                xy=[np.column_stack(((ring-center)@u,(ring-center)@v)) for ring in cleaned]
                poly=Polygon(xy[0],[p for p in xy[1:] if len(p)>=3])
                if poly.is_empty or poly.area<1e-5:
                    log['degenerate_omitted']+=1;continue
                if not poly.is_valid:
                    log['invalid_ring_omitted']+=1
                    rejected.append(dict(object=key,surface=surf_idx,reason='invalid_ring'))
                    continue
                if planarity>.02:
                    log['nonplanar_over_20mm_omitted']+=1
                    rejected.append(dict(object=key,surface=surf_idx,reason='planarity',deviation_m=float(planarity),area=float(poly.area)))
                    continue
                source_area+=poly.area
                if len(poly.interiors):log['surfaces_with_holes_triangulated']+=1
                pieces=convex_pieces(poly)
                assert abs(sum(Polygon(p).area for p in pieces)-poly.area)<max(1e-5,poly.area*1e-7)
                log['accepted_source_surfaces']+=1
                if len(pieces)>1:log['nonconvex_decomposed']+=1
                for piece in pieces:
                    v3=center+piece[:,0,None]*u+piece[:,1,None]*v
                    kind='wall' if semantic=='WallSurface' else ('flat_roof' if normal[2]>.999 else 'pitched_roof')
                    p=core.old.GENERATOR.make_polygon(len(faces)+1,bid,'building',kind,v3,normal,True,True,
                        metadata=dict(dataset=name,source_object=key,source_surface=surf_idx,planarity_m=float(planarity)))
                    faces.append(p)
    report=dict(dataset=name,lod=str(lod),origin_m=origin.tolist(),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        counts=dict(log),accepted_area_m2=source_area,output_convex_faces=len(faces),
        building_count=len({p.object_id for p in faces}),planarity_max_m=max(deviations),
        planarity_p95_m=float(np.percentile(deviations,95)),rejected=rejected,preparation_s=time.perf_counter()-started,
        extent_local=[*np.min(vertices,axis=0),*np.max(vertices,axis=0)])
    profile=SimpleNamespace(casters=faces,receivers=faces,stats={})
    bundle=SimpleNamespace(split='external',terrain=name,seed=20260922,scene=SimpleNamespace(all_polygons=faces,area_size=float(np.ptp(vertices[:,:2],axis=0).max()),n_buildings=report['building_count']),
        raw_profile=profile,proxy_profile=profile,receivers={},report=report)
    core.old.atomic_json(ROOT/'results'/f'{name}_geometry_audit.json',report)
    return bundle

def select_stratified(bundle,per_kind,seed):
    """Random within orientation/height bins, one face per building per type."""
    rng=np.random.default_rng(seed); receivers={}; report={}
    for kind in ('wall','roof'):
        eligible=[p for p in bundle.raw_profile.receivers if p.object_kind=='building' and p.area>=2 and (p.surface_kind=='wall' if kind=='wall' else 'roof' in p.surface_kind)]
        if bundle.split=='external':
            ex=bundle.report['extent_local']
            # Fixed 250 m context margin; all-tile exact baseline remains explicit.
            eligible=[p for p in eligible if ex[0]+250<=p.center[0]<=ex[3]-250 and ex[1]+250<=p.center[1]<=ex[4]-250]
        if len(eligible)<per_kind: raise ValueError((bundle.terrain,kind,'too few eligible',len(eligible)))
        heights=np.array([p.center[2] for p in eligible]); cut=np.median(heights)
        bins={}
        for p in eligible:
            orientation=int(core.old.horizontal_azimuth(p.normal)//90) if kind=='wall' else int(p.normal[2]<.99)
            key=(orientation,int(p.center[2]>cut))
            bins.setdefault(key,[]).append(p)
        for key in bins:rng.shuffle(bins[key])
        keys=sorted(bins); chosen=[]; buildings=set()
        while len(chosen)<per_kind:
            progress=False
            for key in keys:
                while bins[key]:
                    p=bins[key].pop()
                    if p.object_id not in buildings:
                        chosen.append(p);buildings.add(p.object_id);progress=True;break
                if len(chosen)==per_kind:break
            if not progress:break
        if len(chosen)<per_kind:raise ValueError((bundle.terrain,kind,'too few distinct buildings',len(chosen)))
        for i,p in enumerate(chosen):receivers[f'{kind}_{i:02d}']=p
        report[kind]=dict(eligible=len(eligible),selected=len(chosen),distinct_buildings=len(buildings))
    bundle.receivers=receivers
    return report

def main():
    ROOT.joinpath('results/scenes').mkdir(exist_ok=True,parents=True)
    bundles=core.synthetic_bundles()
    for name,file,lod in [('delft_3dbag','3dbag_9-284-556.city.json','2.2'),('rotterdam_delfshaven','rotterdam_delfshaven.city.json','2')]:
        out=ROOT/'results/scenes'/f'{name}_geometry.pkl'
        if out.exists():
            with out.open('rb') as f:b=pickle.load(f)
        else:
            b=cityjson_surfaces(ROOT/'data'/file,name,lod);core.old.atomic_pickle(out,b)
        bundles.append(b)
    manifest=[]
    for i,b in enumerate(bundles):
        per_kind=4 if b.split=='calibration' else (32 if b.split=='external' else 8)
        selection=select_stratified(b,per_kind,20260922+i)
        key=f'{b.split}__{b.terrain}__{b.seed}'
        core.old.atomic_pickle(ROOT/'results/scenes'/f'{key}.pkl',b)
        row=dict(key=key,split=b.split,terrain=b.terrain,seed=b.seed,selection=selection,
            raw_casters=len(b.raw_profile.casters),proxy_casters=len(b.proxy_profile.casters),
            receivers=[dict(label=k,sid=p.sid,building=p.object_id,kind=p.surface_kind,area_m2=p.area,normal=p.normal,center=p.center) for k,p in b.receivers.items()])
        manifest.append(row); print(key,len(b.receivers),len(b.raw_profile.casters),flush=True)
    core.old.atomic_json(ROOT/'results/frozen_sample_manifest.json',dict(seed=20260922,created=core.old.now_iso(),selection_before_outcomes=True,scenes=manifest))

if __name__=='__main__':main()
