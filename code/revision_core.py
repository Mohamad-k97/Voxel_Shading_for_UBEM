"""Corrected, testable geometry and evaluation primitives for the R1 study.

The submitted source is imported without alteration. Geometry preparation clips
casters to the receiver's sunward half-space; no caster-centroid sun test is used.
"""
from __future__ import annotations
import sys, math, time, pickle, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numba import njit
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import union_all
from shapely.errors import GEOSException

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'source/original_package/urban_shading_method_article'
sys.path.insert(0, str(PACKAGE / 'code'))
import benchmark_study as old

EPS = 1e-9

@njit(cache=True)
def clip_plane(points):
    """Clip local (u,v,n) coordinates to n>=1e-9, retaining edge intersections."""
    out = np.empty((len(points)+2,3))
    count = 0
    a = points[-1]
    for b in points:
        ia, ib = a[2] >= EPS, b[2] >= EPS
        if ia != ib:
            t = (EPS-a[2])/(b[2]-a[2])
            out[count] = a + t*(b-a)
            count += 1
        if ib:
            out[count] = b
            count += 1
        a = b
    return out[:count].copy()

@njit(cache=True)
def clip_2d(subject, boundary):
    """Sutherland--Hodgman clipping against a convex CCW receiver."""
    cap = len(subject)+len(boundary)+4
    a = np.empty((cap,2)); b = np.empty((cap,2))
    n = len(subject)
    a[:n] = subject
    for j in range(len(boundary)):
        if n < 3:
            return a[:0].copy()
        p, q = boundary[j-1], boundary[j]
        ex, ey = q[0]-p[0], q[1]-p[1]
        prev = a[n-1].copy()
        fp = ex*(prev[1]-p[1])-ey*(prev[0]-p[0])
        k = 0
        for i in range(n):
            cur = a[i].copy()
            fc = ex*(cur[1]-p[1])-ey*(cur[0]-p[0])
            if (fp >= -EPS) != (fc >= -EPS):
                t = fp/(fp-fc)
                b[k] = prev+t*(cur-prev)
                k += 1
            if fc >= -EPS:
                b[k] = cur
                k += 1
            prev, fp = cur, fc
        a,b = b,a
        n = k
    return a[:n].copy()

@njit(cache=True)
def area_2d(p):
    s=0.
    for i in range(len(p)):
        s += p[i-1,0]*p[i,1]-p[i,0]*p[i-1,1]
    return abs(s)*.5

@njit(cache=True)
def projected_clips(vertices, counts, boundary, du, dv, dn):
    clips = []
    ids = []
    areas = []
    if dn <= 1e-9:
        return clips, ids, areas
    xmin,xmax = np.min(boundary[:,0]),np.max(boundary[:,0])
    ymin,ymax = np.min(boundary[:,1]),np.max(boundary[:,1])
    for i in range(len(counts)):
        n = counts[i]
        if n < 3: continue
        p = np.empty((n,2))
        for j in range(n):
            p[j,0] = vertices[i,j,0]-vertices[i,j,2]*du/dn
            p[j,1] = vertices[i,j,1]-vertices[i,j,2]*dv/dn
        if np.max(p[:,0])<xmin-EPS or np.min(p[:,0])>xmax+EPS or np.max(p[:,1])<ymin-EPS or np.min(p[:,1])>ymax+EPS:
            continue
        c = clip_2d(p,boundary)
        area = area_2d(c)
        if area > EPS:
            clips.append(c); ids.append(i); areas.append(area)
    return clips,ids,areas

@njit(cache=True)
def sample_single(vertices, boundary, directions, normal, u, v, area):
    result = np.zeros(len(directions))
    for k in range(len(directions)):
        d=directions[k]; dn=np.dot(normal,d)
        if dn<=1e-9 or len(vertices)<3: continue
        du,dv=np.dot(u,d),np.dot(v,d)
        p=np.empty((len(vertices),2))
        for j in range(len(vertices)):
            p[j,0]=vertices[j,0]-vertices[j,2]*du/dn
            p[j,1]=vertices[j,1]-vertices[j,2]*dv/dn
        result[k]=min(1.,area_2d(clip_2d(p,boundary))/area)
    return result

def directions(az,el):
    a,e=np.radians(az),np.radians(el)
    return np.column_stack((np.cos(e)*np.sin(a),np.cos(e)*np.cos(a),np.sin(e)))

@dataclass
class PackedGeometry:
    cache: object
    casters: list
    vertices: np.ndarray
    counts: np.ndarray
    distances: np.ndarray
    sids: np.ndarray
    preparation_s: float

def prepare(receiver, casters):
    start=time.perf_counter()
    c=old.receiver_cache(receiver)
    basis=np.column_stack((c.basis_u,c.basis_v,c.normal))
    local=[clip_plane(np.ascontiguousarray((p.vertices-c.origin)@basis)) for p in casters]
    max_n=max([len(p) for p in local]+[3])
    verts=np.zeros((len(local),max_n,3))
    counts=np.array([len(p) for p in local],dtype=np.int64)
    for i,p in enumerate(local): verts[i,:len(p)]=p
    return PackedGeometry(c,list(casters),verts,counts,
        np.array([old.horizontal_distance(receiver,p) for p in casters]),
        np.array([p.sid for p in casters]),time.perf_counter()-start)

def exact_parts(g,d):
    c=g.cache
    clips,idx,areas=projected_clips(g.vertices,g.counts,c.polygon_2d,
        float(c.basis_u@d),float(c.basis_v@d),float(c.normal@d))
    polys=[];valid_idx=[];valid_areas=[]
    for points,i,area in zip(clips,idx,areas):
        p=Polygon(points)
        if not p.is_valid:
            repaired=p.buffer(0)
            # Only roundoff-level topology repairs are accepted automatically.
            if abs(repaired.area-area)>max(1e-7,c.area*1e-8):
                # Nearly tangent edges can form an invalid zero-width spike.
                # Independently re-intersect the valid projected source and
                # convex receiver. Require area agreement with the original
                # shoelace result rather than accepting a destructive buffer.
                vertices=g.vertices[i,:g.counts[i]]
                dn=float(c.normal@d)
                source=Polygon(np.column_stack((vertices[:,0]-vertices[:,2]*float(c.basis_u@d)/dn,
                                                 vertices[:,1]-vertices[:,2]*float(c.basis_v@d)/dn)))
                if not source.is_valid:raise ValueError(('invalid projected source',g.casters[i].sid))
                repaired=source.intersection(Polygon(c.polygon_2d))
                if not repaired.is_valid or abs(repaired.area-area)>max(1e-7,c.area*1e-8):
                    raise ValueError(('material clipping topology repair',g.casters[i].sid,area,repaired.area))
            p=repaired
        if not p.is_empty and p.area>EPS:
            polys.append(p);valid_idx.append(i);valid_areas.append(p.area)
    return polys,np.asarray(valid_idx,dtype=int),np.asarray(valid_areas)/c.area

def fraction(polys,area):
    if not polys: return 0.
    if len(polys)==1: return float(np.clip(polys[0].area/area,0,1))
    if max(p.area for p in polys)>=area*(1-1e-10): return 1.
    try:
        value=unary_union(polys).area/area
    except GEOSException:
        # GEOS floating overlay can fail on coincident valid edges. Verify
        # two nanometre-scale snap grids agree before accepting a robust union.
        fine=union_all(polys,grid_size=1e-8).area/area
        coarse=union_all(polys,grid_size=1e-7).area/area
        if abs(fine-coarse)>1e-6:raise ValueError(('union precision instability',fine,coarse))
        value=fine
    return float(np.clip(value,0,1))

def exact(g,az,el):
    p,_,_=exact_parts(g,old.sun_vector(az,el))
    return fraction(p,g.cache.area)

def corrected_single(cache,caster,az,el):
    basis=np.column_stack((cache.basis_u,cache.basis_v,cache.normal))
    local=clip_plane(np.ascontiguousarray((caster.vertices-cache.origin)@basis))
    return float(sample_single(local,cache.polygon_2d,directions(np.array([az]),np.array([el])),
        cache.normal,cache.basis_u,cache.basis_v,cache.area)[0])

def caster_training(cache,caster):
    support=old.angular_support(cache,caster)
    da=max(.75,support['azimuth_max_rel_deg']-support['azimuth_min_rel_deg'])
    de=max(.75,support['elevation_max_deg']-support['elevation_min_deg'])
    aa=support['geometric_azimuth_deg']+np.linspace(support['azimuth_min_rel_deg']-max(.25,.15*da),support['azimuth_max_rel_deg']+max(.25,.15*da),5)
    ee=np.linspace(max(2.,support['elevation_min_deg']-max(.25,.15*de)),min(88.,support['elevation_max_deg']+max(.25,.15*de)),5)
    a,e=np.meshgrid(aa%360,ee)
    ag,eg=np.meshgrid(old.FIT_GLOBAL_AZIMUTHS_DEG,old.FIT_GLOBAL_ELEVATIONS_DEG)
    dg=directions(ag.ravel(),eg.ravel()); valid=dg@cache.normal>.02
    az=np.r_[a.ravel(),ag.ravel()[valid]]; el=np.r_[e.ravel(),eg.ravel()[valid]]
    basis=np.column_stack((cache.basis_u,cache.basis_v,cache.normal))
    local=clip_plane(np.ascontiguousarray((caster.vertices-cache.origin)@basis))
    y=sample_single(local,cache.polygon_2d,directions(az,el),cache.normal,cache.basis_u,cache.basis_v,cache.area)
    return az,el,y,support

def fit_receiver(receiver,casters,path,kernels=('supergaussian',)):
    """All candidates sampled and fitted before rank filtering; restartable."""
    c=old.receiver_cache(receiver); path=Path(path)
    sids=np.asarray([p.sid for p in casters]); n=len(casters)
    pars={k:np.full((n,5),np.nan) for k in kernels}
    peak=np.full(n,np.nan); mass=peak.copy(); score=peak.copy(); elapsed=peak.copy()
    if path.exists():
        with np.load(path) as z:
            assert np.array_equal(z['sids'],sids)
            for k in kernels:
                if k in z:pars[k][:]=z[k]
            for name,arr in [('peak',peak),('mass',mass),('score',score),('elapsed',elapsed)]: arr[:]=z[name]
    def save():
        old.atomic_npz(path,sids=sids,peak=peak,mass=mass,score=score,elapsed=elapsed,**pars)
    for i,p in enumerate(casters):
        if np.isfinite(peak[i]) and all(np.all(np.isfinite(pars[k][i])) for k in kernels):continue
        start=time.perf_counter()
        # Conservative geometry-only zero tests, valid for above-horizon rays.
        impossible=np.max((p.vertices-c.origin)@c.normal)<=EPS or np.max(p.vertices[:,2])<=np.min(receiver.vertices[:,2])+EPS
        if impossible:
            aa,ee=old.vector_azimuth_elevation(p.center-c.origin)
            for k in kernels:pars[k][i]=[aa,np.clip(ee,2,88),1,1,0]
            peak[i]=0.;mass[i]=0.
        else:
            az,el,y,support=caster_training(c,p)
            for k in kernels: pars[k][i]=old.fit_kernel(k,az,el,y,support)
            peak[i]=y.max(); mass[i]=y[25:].sum()
        score[i]=p.area/max(1e-10,float(np.sum((p.center-c.origin)**2)))
        elapsed[i]=time.perf_counter()-start
        if (i+1)%50==0: save()
    save()
    rank=np.lexsort((sids,-score,-mass,-peak))
    return dict(params=pars,sids=sids,peak=peak,mass=mass,score=score,elapsed=elapsed,rank=rank)

def predict(params,az,el,gamma=.6,kernel='supergaussian'):
    az=np.atleast_1d(az); el=np.atleast_1d(el)
    p=np.asarray(params)
    if len(p)==0:return np.zeros(len(az))
    q=((old.wrap_180(az[:,None]-p[None,:,0]))/np.maximum(.05,p[None,:,2]))**2+((el[:,None]-p[None,:,1])/np.maximum(.05,p[None,:,3]))**2
    if kernel=='supergaussian': val=p[None,:,4]*np.exp(-.5*q*q)
    elif kernel=='biweight':val=p[None,:,4]*np.where(q<1,(1-q)**2,0)
    else:
        val=np.array([old.kernel_values_many(kernel,a,e,p) for a,e in zip(az,el)])
    h=-np.expm1(gamma*np.log1p(-np.clip(val,0,1-1e-12)).sum(axis=1))
    return np.maximum(val.max(axis=1),np.minimum(np.minimum(1.,val.sum(axis=1)),h))

def bounded_union(values,gamma):
    values=np.clip(np.asarray(values,dtype=float),0,1)
    if len(values)==0:return 0.
    h=-np.expm1(gamma*np.log1p(-np.minimum(values,1-1e-12)).sum())
    return float(max(values.max(),min(1.,values.sum(),h)))

def bounded_lobe_scalar(kernel,params,az,el,gamma):
    return bounded_union(old.kernel_values_many(kernel,az,el,params),gamma)

def union_tests():
    cases=[]
    for gamma in [.3,.7,1.,1.2]:
        for values in [[],[0.],[.5],[1.],[.5,.5],[.2,.3,.4]]:
            out=bounded_union(values,gamma);lo=max(values) if values else 0.;hi=min(1.,sum(values))
            assert lo-1e-12<=out<=hi+1e-12
            if len(values)==1:assert abs(out-values[0])<1e-12
            cases.append(dict(gamma=gamma,marginals=values,result=out,lower=lo,upper=hi))
    old.atomic_json(ROOT/'results/union_bound_tests.json',dict(passed=len(cases),cases=cases))

def uniform_receiver_points(receiver,n=1024):
    """Deterministic area-weighted quasi-Monte Carlo fan sampling (convex face)."""
    from scipy.stats import qmc
    vertices=np.asarray(receiver.vertices)
    triangles=np.asarray([[vertices[0],vertices[i],vertices[i+1]] for i in range(1,len(vertices)-1)])
    weights=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    seq=qmc.Sobol(3,scramble=False).random_base2(int(math.ceil(math.log2(n))))[:n]
    which=np.minimum(np.searchsorted(np.cumsum(weights/weights.sum()),seq[:,0],side='right'),len(triangles)-1)
    tri=triangles[which]; a=np.sqrt(seq[:,1]); b=seq[:,2]
    return (1-a[:,None])*tri[:,0]+(a*(1-b))[:,None]*tri[:,1]+(a*b)[:,None]*tri[:,2]

def ray_engine(casters):
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    start=time.perf_counter()
    triangles=[]
    for p in casters:
        v=p.vertices
        triangles.extend([[v[0],v[j],v[j+1]] for j in range(1,len(v)-1)])
    triangles=np.asarray(triangles)
    mesh=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
    eng=RayMeshIntersector(mesh)
    # Force BVH construction into preprocessing, including one harmless ray.
    eng.intersects_any([mesh.bounds[1]+1],[[1,0,0]])
    return eng,time.perf_counter()-start,mesh

def ray_fractions(engine,receiver,az,el,n=1024,batch=24):
    pts=uniform_receiver_points(receiver,n)
    dd=directions(np.asarray(az),np.asarray(el)); out=[]
    for start in range(0,len(dd),batch):
        d=dd[start:start+batch]; origins=np.tile(pts,(len(d),1))+np.repeat(d,n,axis=0)*1e-6
        hits=engine.intersects_any(origins,np.repeat(d,n,axis=0)).reshape(len(d),n)
        out.extend(hits.mean(axis=1))
    return np.array(out)

def synthetic_bundles():
    class SourceUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module=='__main__' and name=='SceneBundle': return old.SceneBundle
            return super().find_class(module,name)
    result=[]
    for p in sorted((PACKAGE/'study_output/scenes').glob('*/scene_bundle.pkl')):
        with p.open('rb') as f: result.append(SourceUnpickler(f).load())
    return result

def make_poly(sid,vertices,normal=(1,0,0),object_id=1):
    return old.GENERATOR.make_polygon(sid,object_id,'building','wall',np.array(vertices,dtype=float),np.array(normal,dtype=float),True,True)

def self_test():
    r=make_poly(1,[[0,-5,0],[0,5,0],[0,5,10],[0,-5,10]])
    front=make_poly(2,[[1,-5,0],[1,5,0],[1,5,4],[1,-5,4]],object_id=2)
    rear=make_poly(3,[[-1,-5,0],[-1,5,0],[-1,5,4],[-1,-5,4]],object_id=3)
    cross=make_poly(4,[[-1,-5,4],[1,-5,4],[1,5,4],[-1,5,4]],normal=(0,0,1),object_id=4)
    cases=[]
    for name,cs,az,el,expected in [('finite_receiver_centroid_counterexample',[front],90,45,.3),('rear_caster_rejected',[rear],90,45,0.),('partial_halfspace_crossing',[cross],90,45,.1),('duplicate_shadow_union',[front,front],90,45,.3),('normal_incidence',[front],90,0,.4),('backface',[front],270,45,0.)]:
        value=exact(prepare(r,cs),az,el)
        assert abs(value-expected)<1e-8,(name,value,expected)
        cases.append(dict(test=name,expected=expected,actual=value))
    offset=np.array([1e6,-2e6,100.])
    rt=make_poly(1,r.vertices+offset); ct=make_poly(2,front.vertices+offset,object_id=2)
    translated=exact(prepare(rt,[ct]),90,45)
    assert abs(translated-.3)<1e-8
    cases.append(dict(test='translation_invariance',expected=.3,actual=translated))
    eng,_,_=ray_engine([front])
    ray=ray_fractions(eng,r,[90],[45],16384)[0]
    assert abs(ray-.3)<.003
    cases.append(dict(test='independent_Embree_counterexample',expected=.3,actual=ray))
    old.atomic_json(ROOT/'results/reference_unit_tests.json',dict(passed=len(cases),cases=cases))
    print(json.dumps(cases,indent=2),flush=True)

if __name__=='__main__': self_test()
