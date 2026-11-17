"""Use the actual float32 archive-header normal in every packed prediction.

No fits, geometry, sample selection or calibration parameters are changed.
"""
import json
import numpy as np
import revision_core as c
from run_revision import RESULTS,bundles,error

settings=json.loads((RESULTS/'calibration_final.json').read_text())
changes=[]
for item,b in bundles():
    if b.split=='calibration':continue
    for label,r in b.receivers.items():
        directory=RESULTS/'validation'/item['key']/label
        path=directory/'grid.npz'
        if not path.exists():continue
        z=dict(np.load(path));normal=float(np.float32(c.old.receiver_cache(r).normal_azimuth_deg))
        pars=c.old.dequantize_params(z['packed'],normal)
        pred=c.predict(pars,z['az'],z['el'],settings['gamma'],settings['kernel'])
        changes.append(float(np.max(np.abs(pred-z['values'][:,6]))))
        z['values'][:,6]=pred;c.old.atomic_npz(path,**z)
        hpath=directory/'hourly.npz'
        if hpath.exists():
            h=dict(np.load(hpath));mask=h['front_daylight']
            h['shade'][mask,1]=c.predict(pars,h['az'][mask],h['el'][mask],settings['gamma'],settings['kernel'])
            c.old.atomic_npz(hpath,**h)
            mp=directory/'hourly_meta.json';m=json.loads(mp.read_text());m.update(error(h['shade'][mask,0],h['shade'][mask,1]));c.old.atomic_json(mp,m)
c.old.atomic_json(RESULTS/'archive_header_precision_check.json',dict(receivers=len(changes),maximum_prediction_change=max(changes),normal_header_dtype='float32',scientific_selection_unchanged=True))
print('Archive precision harmonised:',len(changes),'maximum change',max(changes),flush=True)
