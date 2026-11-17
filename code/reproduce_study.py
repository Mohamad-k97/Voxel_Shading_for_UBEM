"""Non-destructive cached verification or a fresh, separately located rerun."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(root,script,*args):
    subprocess.run([sys.executable,str(root/'code'/script),*args],check=True,cwd=root)

def check():
    import numpy as np
    import revision_core as c
    manifest=ROOT/'FILE_MANIFEST.json'
    if manifest.exists():
        for row in json.loads(manifest.read_text())['files']:
            p=ROOT/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
    c.self_test();c.union_tests()
    paths=list((ROOT/'final_results/validation').glob('*/*/grid.npz'))
    assert len(paths)==256
    totals={False:[0,0,0.],True:[0,0,0.]}
    for p in paths:
        with np.load(p) as z:
            raw=z['values'][:,0];est=z['values'][:,6]
            active=raw>.01;acc=totals[p.parents[1].name.startswith('external')]
            acc[0]+=len(raw);acc[1]+=int(active.sum());acc[2]+=float(np.abs(raw[active]-est[active]).sum())
    assert sum(x[0] for x in totals.values())==117051
    assert sum(x[1] for x in totals.values())==23600
    expected={False:.0826245,True:.2018884}
    for external,values in totals.items():assert abs(values[2]/values[1]-expected[external])<1e-6
    assert len(list((ROOT/'final_results/validation').glob('*/*/hourly.npz')))==64
    assert len(list((ROOT/'final_results/validation').glob('*/*/comparison.json')))==32
    summaries=list((ROOT/'final_results/thermal').glob('*/*/*/summary.json'));assert len(summaries)==64
    for p in summaries:
        s=json.loads(p.read_text());assert s['warnings']==0
    print(json.dumps(dict(grid_receivers=256,queries=117051,active=23600,hourly_receivers=64,comparison_receivers=32,thermal_runs=64,all_checks_passed=True),indent=2))

def fresh(destination):
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise SystemExit('Refusing to overwrite an existing destination.')
    ep=list((ROOT/'dependencies').glob('EnergyPlus-*/energyplus'))
    if not ep:raise SystemExit('Install the official EnergyPlus distribution under dependencies first.')
    destination.mkdir(parents=True)
    for name in ['code','data','source']:
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.tar.gz','energyplus_io_reference.txt'))
    (destination/'dependencies').mkdir()
    (destination/'dependencies'/ep[0].parent.name).symlink_to(ep[0].parent,target_is_directory=True)
    run(destination,'revision_core.py')
    subprocess.run([sys.executable,'-c',"import sys; sys.path.insert(0,'code'); import revision_core as c; c.union_tests()"],cwd=destination,check=True)
    run(destination,'prepare_scenes.py')
    source=destination/'results';final=destination/'final_results';final.mkdir()
    shutil.copytree(source/'scenes',final/'scenes')
    for p in source.glob('*.json'):shutil.copy2(p,final/p.name)
    for script,args in [
        ('run_revision.py',['all']),('additional_experiments.py',['freeze']),
        ('additional_experiments.py',['hourly']),('analyse_results.py',['primary']),
        ('thermal_study.py',['study']),('additional_experiments.py',['compare']),
        ('analyse_results.py',['secondary']),('extended_diagnostics.py',[]),
        ('report_values.py',[]),('make_figures.py',[])]:run(destination,script,*args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','fresh']);p.add_argument('destination',nargs='?');a=p.parse_args()
    if a.mode=='fresh':
        if not a.destination:p.error('fresh requires a new destination')
        fresh(a.destination)
    else:check()
