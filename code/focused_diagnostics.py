"""Focused summaries of the frozen results and a stored-caster illustration.

No fitting, calibration, geometry selection, or timing measurement is rerun.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'final_results'
OUT = ROOT / 'focused_analysis'
OUT.mkdir(exist_ok=True)


def marginals(params, az, el):
    q = ((((az[:, None] - params[None, :, 0] + 180) % 360 - 180)
          / np.maximum(.05, params[None, :, 2])) ** 2
         + ((el[:, None] - params[None, :, 1]) / np.maximum(.05, params[None, :, 3])) ** 2)
    return params[None, :, 4] * np.where(q < 1, (1 - q) ** 2, 0)


def aggregate(values):
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    h = -np.expm1(.6 * np.log1p(-np.clip(values, 0, 1-1e-12)).sum(axis=1))
    return np.maximum(values.max(axis=1), np.minimum(np.minimum(1, values.sum(axis=1)), h))


def main():
    d = pd.read_csv(DATA / 'analysis/directional_results.csv.gz')
    active = d[(d.split == 'external') & (d.raw > .01)].copy()
    rows = []
    for patch, g in active.groupby('terrain'):
        row = {'patch': patch, 'active_n': len(g), 'receivers': g.id.nunique()}
        for name, a, b in [('radius', 'raw', 'radius'), ('rank', 'radius', 'topk'),
                           ('aggregation', 'topk', 'exact_marginal_union'),
                           ('fitting', 'exact_marginal_union', 'float'),
                           ('quantisation', 'float', 'packed')]:
            row[name + '_signed_bias'] = float((g[b] - g[a]).mean())
        row['final_signed_bias'] = float((g.packed-g.raw).mean())
        assert np.isclose(sum(row[n+'_signed_bias'] for n in
                             ['radius', 'rank', 'aggregation', 'fitting', 'quantisation']),
                          row['final_signed_bias'], atol=1e-12)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'external_common_cohort_bias.csv', index=False)

    sweep = pd.read_csv(DATA/'analysis/exploratory_K_sensitivity.csv')
    caprows = []
    for (patch, k), g in sweep[sweep.split=='external'].groupby(['terrain', 'K']):
        caprows.append({'patch':patch, 'K':int(k), 'receivers':len(g),
                        'active_n':int(g.active_n.sum()),
                        'active_mae':float(np.average(g.active_mae, weights=g.active_n)),
                        'active_bias':float(np.average(g.active_bias, weights=g.active_n)),
                        'payload_bytes':int(g.fixed_record_bytes.iloc[0])})
    pd.DataFrame(caprows).to_csv(OUT/'external_capacity_by_patch.csv', index=False)

    examples = []
    for patch in ['delft_3dbag', 'rotterdam_delfshaven']:
        scene = 'external__' + patch + '__20260922'
        directory = DATA/'validation'/scene/'wall_00'
        z = np.load(directory/'grid.npz')
        meta = json.loads((directory/'metadata.json').read_text())
        n = np.array(meta['normal'])
        normal_az = float(np.float32(np.degrees(np.arctan2(n[0],n[1])) % 360))
        p = z['packed'].astype(float)
        p[:, :4] /= 10
        p[:, 0] = (p[:, 0] + normal_az) % 360
        p[:, 4] /= 1000
        values = marginals(p,z['az'],z['el'])
        full = aggregate(values)
        assert np.allclose(full,z['values'][:,6],atol=2e-12,rtol=1e-10)
        # Demonstrate removal of the retained surface with greatest integrated
        # marginal response over this receiver's stored validation directions.
        j = int(np.argmax(values.sum(axis=0)))
        reduced = aggregate(np.delete(values,j,axis=1))
        assert np.all(reduced <= full+1e-12)
        i = int(np.argmax(full-reduced))
        example = {'patch':patch, 'scene':scene, 'receiver':'wall_00',
                   'receiver_sid':meta['sid'], 'caster_sid':int(z['selected_sids'][j]),
                   'retained_before':len(p), 'retained_after':len(p)-1,
                   'azimuth_deg':float(z['az'][i]), 'elevation_deg':float(z['el'][i]),
                   'original_estimate':float(full[i]), 'removed_caster_marginal':float(values[i,j]),
                   'estimate_after_removal':float(reduced[i]),
                   'maximum_estimated_change':float(full[i]-reduced[i]),
                   'original_prediction_reproduced_max_difference':float(np.max(np.abs(full-z['values'][:,6]))),
                   'scope':'Illustration of existing archive operations; no new geometric accuracy or timing claim.'}
        examples.append(example)
        np.savez_compressed(OUT/(patch+'_caster_removal.npz'),params=p,caster_sids=z['selected_sids'],
                            az=z['az'],el=z['el'],original=full,removed=reduced,
                            removed_index=np.array([j]))
    (OUT/'caster_selectivity_examples.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps({'bias_attribution':rows,'caster_examples':examples},indent=2))


if __name__=='__main__':
    main()
