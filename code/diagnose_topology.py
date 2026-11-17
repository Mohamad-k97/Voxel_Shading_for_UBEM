import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import revision_core as c
from run_revision import bundles,angle_grid
for item,b in bundles('external'):
    if b.terrain!='rotterdam_delfshaven':continue
    r=b.receivers['roof_28'];g=c.prepare(r,c.old.building_casters(b.raw_profile,r));a,e=angle_grid(r)
    print('RECEIVER',r.sid,r.area,r.vertices,r.normal,flush=True)
    for az,el,d in zip(a,e,c.directions(a,e)):
        cache=g.cache;du=float(cache.basis_u@d);dv=float(cache.basis_v@d);dn=float(cache.normal@d)
        clips,ids,areas=c.projected_clips(g.vertices,g.counts,cache.polygon_2d,du,dv,dn)
        for pts,idx,area in zip(clips,ids,areas):
            poly=Polygon(pts)
            if not poly.is_valid and abs(poly.buffer(0).area-area)>max(1e-7,cache.area*1e-8):
                v=g.vertices[idx,:g.counts[idx]];proj=np.c_[v[:,0]-v[:,2]*du/dn,v[:,1]-v[:,2]*dv/dn]
                p=Polygon(proj);clip=p.intersection(Polygon(cache.polygon_2d))
                print('BAD',az,el,g.sids[idx],area,poly.buffer(0).area,explain_validity(poly))
                print('CLIPPED',pts,'SOURCE',proj,'SOURCE_VALID',p.is_valid,'GEOS',clip.wkt,'GEOSAREA',clip.area,flush=True)
                raise SystemExit
