"""Scientific figures regenerated from the frozen revision outputs."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import revision_core as c
from run_revision import RESULTS,bundles
from analyse_results import LABELS

OUT=c.ROOT/'figures';OUT.mkdir(exist_ok=True)
NAVY='#263e55';TEAL='#247f79';ORANGE='#bb6335';GRAY='#66717c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240,'axes.labelcolor':'#222222','text.color':'#222222'})

def save(fig,n):
    fig.savefig(OUT/f'figure_{n:02d}.png',dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.08);plt.close(fig)
def pooled(df,col='active_mae'):
    return float(np.nansum(df[col]*df.active_n)/df.active_n.sum())

def concept():
    # Explicit teaching geometry, not an error-selected validation observation.
    r=c.make_poly(1,[[0,-5,0],[0,5,0],[0,5,10],[0,-5,10]])
    caster=c.make_poly(2,[[10,-3,8],[10,3,8],[10,3,16],[10,-3,16]],object_id=2)
    az,el,y,support=c.caster_training(c.old.receiver_cache(r),caster)
    pars=c.old.fit_kernel('biweight',az,el,y,support)
    span=min(100,max(15,pars[2]*1.3));azrel=np.linspace(-span,span,80);els=np.linspace(0,90,60)
    aa,ee=np.meshgrid(pars[0]+azrel,els);g=c.prepare(r,[caster])
    truth=np.array([c.exact(g,a,e) if r.normal@d>1e-6 else np.nan for a,e,d in zip(aa.ravel(),ee.ravel(),c.directions(aa.ravel(),ee.ravel()))]).reshape(aa.shape)
    approx=c.predict(pars[None,:],aa.ravel(),ee.ravel(),.6,'biweight').reshape(aa.shape);approx[np.isnan(truth)]=np.nan
    fig=plt.figure(figsize=(6.3,4.7),layout='constrained');gs=fig.add_gridspec(2,2)
    ax=fig.add_subplot(gs[0,0],projection='3d');origin=r.center
    for p,color in [(r,TEAL),(caster,ORANGE)]:
        v=p.vertices-origin;ax.add_collection3d(Poly3DCollection([v],facecolors=color,edgecolors=color,alpha=.6,linewidth=.8))
    d=c.old.sun_vector(90,30);polys,_,_=c.exact_parts(g,d)
    for poly in polys:
        uv=np.array(poly.exterior.coords);v=g.cache.origin+uv[:,0,None]*g.cache.basis_u+uv[:,1,None]*g.cache.basis_v-origin
        ax.add_collection3d(Poly3DCollection([v],facecolors=NAVY,edgecolors=NAVY,alpha=.9,linewidth=.8))
    for v in caster.vertices:
        projected=v-d*((v-r.vertices[0])@r.normal)/(d@r.normal)
        ray=np.stack([v,projected])-origin;ax.plot(ray[:,0],ray[:,1],ray[:,2],ls='--',color=GRAY,lw=.6)
    vv=np.vstack((r.vertices,caster.vertices))-origin;lo=vv.min(0);hi=vv.max(0);span3=np.maximum(hi-lo,1)
    ax.set(xlim=(lo[0]-1,hi[0]+1),ylim=(lo[1]-1,hi[1]+1),zlim=(lo[2]-1,hi[2]+1),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]');ax.set_box_aspect(span3)
    ax.view_init(elev=23,azim=-40);ax.set_title('(a) Project and clip a shadow',loc='left')
    ax.set_axis_off()
    ax.text2D(.02,.94,'receiver',color=TEAL,transform=ax.transAxes);ax.text2D(.02,.84,'caster',color=ORANGE,transform=ax.transAxes)
    for grid,position,title in [(truth,gs[0,1],'(b) Exact sampled response'),(approx,gs[1,0],'(c) Five-parameter lobe')]:
        ax=fig.add_subplot(position);im=ax.pcolormesh(azrel,els,grid,vmin=0,vmax=1,cmap='viridis',shading='auto');ax.set(xlabel='Azimuth offset from μφ [°]',ylabel='Elevation [°]',title=title)
        if position==gs[1,0]:
            ax.plot(0,pars[1],'+',color='white',ms=7,mew=1.5)
            ax.annotate('',xy=(min(span,pars[2]),pars[1]),xytext=(0,pars[1]),arrowprops=dict(arrowstyle='<->',color='white',lw=1))
            ax.text(min(span,pars[2])*.4,pars[1]+5,'σφ',color='white')
            ax.annotate('',xy=(0,min(90,pars[1]+pars[3])),xytext=(0,pars[1]),arrowprops=dict(arrowstyle='<->',color='white',lw=1))
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='Shaded fraction')
    ax=fig.add_subplot(gs[1,1]);ax.axis('off');ax.set_title('(d) Store, then evaluate',loc='left')
    lines=[('μφ',f'{pars[0]:.1f}°','azimuth centre'),('μθ',f'{pars[1]:.1f}°','elevation centre'),('σφ',f'{pars[2]:.1f}°','azimuth width'),('σθ',f'{pars[3]:.1f}°','elevation width'),('a',f'{pars[4]:.3f}','amplitude')]
    for i,(symbol,value,meaning) in enumerate(lines):
        y=.88-i*.13;ax.text(.03,y,symbol,color=NAVY,weight='bold');ax.text(.2,y,value,color=NAVY);ax.text(.49,y,meaning)
    ax.text(.03,.12,'Repeat for retained casters; combine their\nvalues within the marginal-union bounds.',va='top',fontsize=8)
    save(fig,1)

def scenes():
    selected=[];seen=set()
    for item,b in bundles():
        if b.split=='calibration' or b.terrain in seen:continue
        seen.add(b.terrain);selected.append((item,b))
    order=['tall_buildings','horizontal_city','small_town','suburban','delft_3dbag','rotterdam_delfshaven'];selected.sort(key=lambda x:order.index(x[1].terrain))
    fig,axs=plt.subplots(3,2,figsize=(6.3,7.1),layout='constrained')
    for letter,ax,(item,b) in zip('abcdef',axs.ravel(),selected):
        faces=[p for p in b.raw_profile.casters if p.object_kind=='building' and 'roof' in p.surface_kind]
        coll=PolyCollection([p.vertices[:,:2] for p in faces],array=np.array([p.center[2] for p in faces]),cmap='cividis',edgecolors='none',linewidths=0)
        ax.add_collection(coll);ax.autoscale();ax.set_aspect('equal');ax.margins(.02)
        for kind,color,marker in [('wall',ORANGE,'o'),('roof',TEAL,'^')]:
            pts=np.array([r.center for label,r in b.receivers.items() if label.startswith(kind)]);ax.scatter(pts[:,0],pts[:,1],s=9,c=color,marker=marker,label=f'{kind} receivers',edgecolors='white',linewidths=.3)
        ax.set_title(f'({letter}) {LABELS[b.terrain]}',loc='left');ax.set(xlabel='Local x [m]',ylabel='Local y [m]');fig.colorbar(coll,ax=ax,fraction=.028,pad=.02,label='Roof z [m]')
    axs[0,0].legend(loc='upper left',framealpha=.85)
    save(fig,3)

def convergence():
    fig,axs=plt.subplots(1,2,figsize=(6.3,2.55),layout='constrained')
    for ax,name,x,unit,selected in [(axs[0],'radius_calibration.csv','radius','Radius [m]',250),(axs[1],'topk_calibration.csv','K','Retained caster limit K',160)]:
        d=pd.read_csv(RESULTS/name).groupby(x)[['active_mae','active_p95']].mean()
        ax.plot(d.index,d.active_mae,'o-',color=TEAL,ms=3,label='Mean active MAE');ax.plot(d.index,d.active_p95,'s-',color=NAVY,ms=3,label='Mean receiver P95')
        ax.axhline(.005,color=TEAL,ls=':',lw=.9);ax.axhline(.02,color=NAVY,ls=':',lw=.9);ax.axvline(selected,color=ORANGE,ls='--',lw=1)
        ax.set(xlabel=unit,ylabel='Shaded-fraction error');ax.set_title('(a) Radius selection' if x=='radius' else '(b) Rank-limit selection',loc='left')
        if x=='K':ax.set_xscale('log',base=2);ax.set_xticks([16,64,160,320,896],['16','64','160','320','896'])
        ax.legend(frameon=False)
    save(fig,4)

def kernels():
    kb=pd.read_csv(RESULTS/'analysis/kernel_comparison.csv').sort_values('mae');g=pd.read_csv(RESULTS/'kernel_gamma_calibration.csv');g=g[g.kernel=='biweight']
    fig,axs=plt.subplots(1,2,figsize=(6.3,3.0),layout='constrained')
    names={'biweight':'Biweight','cosine':'Cosine','gaussian':'Gaussian','wendland':'Wendland','supergaussian':'Super-Gaussian 4','supergaussian6':'Super-Gaussian 6','supergaussian8':'Super-Gaussian 8','logistic':'Logistic'}
    axs[0].barh([names[k] for k in kb.kernel],kb.mae,color=[TEAL]+[GRAY]*7);axs[0].invert_yaxis();axs[0].set(xlabel='Calibration mean receiver active MAE',title='(a) Kernel shape (five parameters)')
    colors=[NAVY,TEAL,ORANGE,'#827250']
    for (terrain,sub),color in zip(g.groupby('terrain'),colors):
        curve=sub.groupby('gamma').active_mae.mean();axs[1].plot(curve.index,curve,color=color,label=LABELS[terrain])
    for kind,ls in [('wall','--'),('roof',':')]:
        curve=g[g.kind==kind].groupby('gamma').active_mae.mean();axs[1].plot(curve.index,curve,color='black',ls=ls,label=kind.capitalize())
    axs[1].axvline(.6,color='#888888',lw=.8);axs[1].set(xlabel='Overlap coefficient γ',ylabel='Mean receiver active MAE',title='(b) Calibration sensitivity');axs[1].legend(frameon=False,fontsize=6.7)
    save(fig,5)

def accuracy():
    d=pd.read_csv(RESULTS/'analysis/directional_results.csv.gz')
    fig,axs=plt.subplots(1,2,figsize=(6.3,2.85),layout='constrained')
    for ax,split,title in zip(axs,['validation','external'],['(a) Synthetic validation','(b) External city geometry']):
        sub=d[(d.split==split)&(d.raw>.01)];im=ax.hist2d(sub.raw,sub.packed,bins=35,range=[[0,1],[0,1]],norm=LogNorm(),cmap='viridis');ax.plot([0,1],[0,1],color=ORANGE,lw=.8)
        ax.set(xlabel='Corrected raw shaded fraction',ylabel='Packed-lobe estimate',title=title);ax.set_aspect('equal');fig.colorbar(im[3],ax=ax,fraction=.04,pad=.02,label='Active query count')
    save(fig,6)

def diagnostics():
    cases=json.loads((RESULTS/'analysis/stage_worst_cases.json').read_text())['cases'];case=next(x for x in cases if x['split']=='synthetic' and x['stage']=='representation')
    z=np.load(RESULTS/'validation'/case['scene']/case['label']/'grid.npz')
    fig,axs=plt.subplots(2,2,figsize=(6.3,5.0),layout='constrained')
    for ax,col,title in [(axs[0,0],3,'(a) Exact retained-caster union'),(axs[0,1],5,'(b) Floating lobe estimate')]:
        grid=np.full((18,36),np.nan);ix=np.rint((z['az']-2.5)/10).astype(int);iy=np.rint((z['el']-2.5)/5).astype(int);grid[iy,ix]=z['values'][:,col]
        im=ax.imshow(grid,origin='lower',extent=(-2.5,357.5,0,90),aspect='auto',cmap='viridis',vmin=0,vmax=1);ax.set(xlabel='Azimuth [°]',ylabel='Elevation [°]',title=title);fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    d=pd.read_csv(RESULTS/'analysis/stratified_error_conditions.csv');d=d[d.stage=='representation']
    for ax,conditions,labels,title in [(axs[1,0],['elevation <10°','elevation 10–30°','elevation 30–60°','elevation ≥60°'],['<10°','10–30°','30–60°','≥60°'],'(c) Solar elevation'),(axs[1,1],['one active retained caster','2–5 active retained casters','≥6 active retained casters'],['1','2–5','≥6'],'(d) Active retained casters')]:
        x=np.arange(len(conditions))
        for offset,split,color in [(-.18,'synthetic',TEAL),(.18,'external',ORANGE)]:
            ss=d[d.split==split].set_index('condition').loc[conditions];ax.bar(x+offset,ss.active_mae,width=.34,color=color,label=split.capitalize())
        ax.set_xticks(x,labels);ax.set(ylabel='Representation active MAE',title=title);ax.legend(frameon=False)
    save(fig,7)

def comparators():
    d=pd.read_csv(RESULTS/'analysis/method_comparison.csv');k=pd.read_csv(RESULTS/'analysis/exploratory_K_sensitivity.csv')
    fig,axs=plt.subplots(1,2,figsize=(6.3,3.15),layout='constrained')
    names={'lobe_biweight':'Lobe K=160','angular_lookup_10deg':'Lookup 10°','angular_lookup_5deg':'Lookup 5°','Embree_256_samples':'Embree 256','Embree_1024_samples':'Embree 1,024','Embree_4096_samples':'Embree 4,096','compiled_polygon_union':'Polygon union'}
    offsets={'lobe_biweight':(7,-12),'angular_lookup_10deg':(4,7),'angular_lookup_5deg':(5,-12),'Embree_256_samples':(-43,10),'Embree_1024_samples':(-40,21),'Embree_4096_samples':(-40,32),'compiled_polygon_union':(-55,-15)}
    for method,ss in d.groupby('method'):
        x=ss.query_s.median()*1e6;y=pooled(ss);color=ORANGE if method=='lobe_biweight' else TEAL if method.startswith('angular') else NAVY
        axs[0].scatter(x,y,c=color,s=23,marker='o');axs[0].annotate(names[method],(x,y),xytext=offsets[method],textcoords='offset points',fontsize=6.5,arrowprops=dict(arrowstyle='-',lw=.4,color=color))
    axs[0].set(xscale='log',xlabel='Median warm query time [μs]',ylabel='Pooled active MAE',title='(a) Measured query–accuracy trade-off');axs[0].margins(x=.4,y=.3)
    for split,color,label in [('validation',TEAL,'Synthetic'),('external',ORANGE,'External')]:
        sub=k[k.split==split];x=[];y=[]
        for cap,ss in sub.groupby('K'):x.append(ss.fixed_record_bytes.iloc[0]/1024);y.append(pooled(ss))
        axs[1].plot(x,y,'o-',color=color,label=label+' lobes',ms=3)
        for xx,yy,cap in zip(x,y,sorted(sub.K.unique())):
            if cap in [32,160,896]:axs[1].annotate(str(cap),(xx,yy),xytext=(3,4),textcoords='offset points',fontsize=6.5,color=color)
        for method,marker in [('angular_lookup_10deg','s'),('angular_lookup_5deg','^')]:
            ss=d[(d.split==split)&(d.method==method)];axs[1].scatter(ss.packed_bytes.iloc[0]/1024,pooled(ss),marker=marker,c=color,s=25)
    axs[1].set(xscale='log',xlabel='Fixed per-receiver payload [KiB]',ylabel='Pooled active MAE',title='(b) Storage sensitivity (exploratory)');axs[1].legend(frameon=False,loc='upper right');axs[1].set_xticks([.5,1,2,4,8,16],['0.5','1','2','4','8','16'])
    save(fig,8)

def consequences():
    h=pd.read_csv(RESULTS/'analysis/hourly_energy_metrics.csv');t=pd.read_csv(RESULTS/'analysis/thermal_differences.csv')
    fig,axs=plt.subplots(1,2,figsize=(6.3,3.15),layout='constrained');rng=np.random.default_rng(9)
    for idx,external in enumerate([False,True]):
        sub=h[h.scene.str.startswith('external')==external];x=idx+rng.uniform(-.16,.16,len(sub));axs[0].scatter(x,sub.beam_bias_kwh_m2,s=17,c=ORANGE if external else TEAL,alpha=.75,edgecolors='white',linewidths=.3)
    axs[0].axhline(0,color=GRAY,lw=.7);axs[0].set_xticks([0,1],['Synthetic','External']);axs[0].set(ylabel='Annual direct-beam energy bias [kWh/m²]',title='(a) Hourly solar trajectories (64 receivers)')
    for external,color,label in [(False,TEAL,'Synthetic'),(True,ORANGE,'External')]:
        sub=t[t.scene.str.startswith('external')==external];axs[1].scatter(sub.delta_cooling_kwh,sub.delta_peak_cooling_w,c=color,s=22,alpha=.75,label=label,edgecolors='white',linewidths=.3)
    axs[1].axhline(0,color=GRAY,lw=.7);axs[1].axvline(0,color=GRAY,lw=.7);axs[1].set(xlabel='Annual cooling-load bias [kWh]',ylabel='Peak cooling-load bias [W]',title='(b) Controlled thermal response (32 walls)');axs[1].legend(frameon=False)
    save(fig,9)

if __name__=='__main__':
    for fn in [concept,scenes,convergence,kernels,accuracy,diagnostics,comparators,consequences]:
        fn();print('FIGURE',fn.__name__,flush=True)
