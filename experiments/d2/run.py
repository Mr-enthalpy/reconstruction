"""Fixed-budget D2 pilot. Explicit input paths; no test tuning or hardware."""
from pathlib import Path
import argparse,json,time,hashlib,csv,sys
import numpy as np
import cv2,torch
from scipy.optimize import minimize
from scipy.special import expit
from lcd_reconstruction.unrolling.spatial_pgd import FFTMeasuredOperator,SpatialSpectralPGD
from lcd_reconstruction.operators.measured_psf import MeasuredPSFOperator

def classify(features,labels,records,train):
    features=np.asarray(features,dtype=float);labels=np.asarray(labels,dtype=int)
    mu=features[train].mean(0);sd=features[train].std(0);sd[sd<1e-9]=1
    x=np.column_stack([np.ones(len(labels)),(features-mu)/sd]);xt=x[train];yt=labels[train]
    count=np.bincount(yt,minlength=2);assert (count>0).all()
    weight=1/count[yt];weight/=weight.sum()
    def loss(w):
        z=xt@w;reg=w.copy();reg[0]=0
        return float((weight*(np.logaddexp(0,z)-yt*z)).sum()+.005*reg@reg),xt.T@(weight*(expit(z)-yt))+.01*reg
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000})
    assert fit.success,fit.message
    pred=expit(x@fit.x)>=.5;result={}
    for split in ['train','val','test']:
        for source in ['all','UWA','CAVE']:
            idx=np.array([r['split']==split and (source=='all' or r['source_dataset']==source) and r['augmentation'] is None for r in records])
            if not idx.any():continue
            cm=np.zeros((2,2),int)
            for t,p in zip(labels[idx],pred[idx]):cm[t,int(p)]+=1
            support=cm.sum(1);present=support>0
            recall=np.divide(cm.diagonal(),support,out=np.zeros(2,float),where=present)
            f1=np.divide(2*cm.diagonal(),cm.sum(0)+support,out=np.zeros(2,float),where=cm.sum(0)+support>0)
            result[split+'_'+source]=dict(n=int(idx.sum()),confusion_matrix=cm.tolist(),accuracy=float((labels[idx]==pred[idx]).mean()),balanced_accuracy=float(recall.mean()) if present.all() else None,macro_f1=float(f1.mean()) if present.all() else None,non_face_recall=float(recall[0]) if support[0] else None,face_recall=float(recall[1]) if support[1] else None)
    return result,[dict(sample_id=r['sample_id'],split=r['split'],source=r['source_dataset'],truth=int(t),pred=int(pr),error=bool(t!=pr)) for r,t,pr in zip(records,labels,pred) if r['augmentation'] is None]

def features_source(x):
    # No metadata labels as features: simple brightness, contrast and spatial statistics.
    return np.array([[a.mean(),a.std(),*np.quantile(a,[.01,.1,.5,.9,.99]),(a==0).mean(),np.abs(np.diff(a.mean(0),axis=0)).mean(),np.abs(np.diff(a.mean(0),axis=1)).mean()] for a in x])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads(a.config.read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    X=np.load(a.prepared/'scenes.npz')['cube'];records=json.loads((a.prepared/'manifest.json').read_text());protocols=json.loads((a.prepared/'protocols.json').read_text());ks=np.load(a.prepared/'kernels.npz');k=ks['k512'];summary={};failures={};checks={};start=time.perf_counter()
    # Verify actual binned measured kernels against original D1 operator.
    xx=rng.normal(size=(21,8,9));yy=rng.normal(size=(8,9));old=MeasuredPSFOperator(k,np.ones(21));pt=FFTMeasuredOperator(torch.tensor(k,dtype=torch.float64),(8,9));
    checks['d1_parity_max_abs']=float(np.max(np.abs(old.forward(xx)-pt(torch.tensor(xx)).numpy())))
    checks['adjoint_relative_error']=float(abs(np.vdot(old.forward(xx),yy)-np.vdot(xx,pt.adjoint(torch.tensor(yy)).numpy()))/max(abs(np.vdot(old.forward(xx),yy)),1e-12));assert checks['d1_parity_max_abs']<1e-9 and checks['adjoint_relative_error']<1e-9
    # Source classifier is restricted to true negative-class content, not face-vs-object labels.
    nidx=[i for i,r in enumerate(records) if r['label']=='non_face' and r['augmentation'] is None]
    nr=[records[i] for i in nidx];nt=np.array([r['split']=='train' for r in nr]);ny=np.array([r['source_dataset']=='CAVE' for r in nr],int)
    summary['source_diagnostic_normalized'],failures['source_diagnostic']=classify(features_source(X[nidx]),ny,nr,nt)
    for metrics in summary['source_diagnostic_normalized'].values():
        metrics['uwa_recall']=metrics.pop('non_face_recall')
        metrics['cave_recall']=metrics.pop('face_recall')
        metrics['class_order']=['UWA','CAVE']
    for protocol,ids in protocols.items():
        x=torch.tensor(X[ids]);rr=[records[i] for i in ids];labels=np.array([r['label']=='face' for r in rr],int)
        orig=np.array([r['augmentation'] is None for r in rr]);train=np.array([r['split']=='train' for r in rr]);train_orig=train&orig
        # Uncentred covariance avoids a hidden affine offset in the physical operator.
        pixels=x[torch.tensor(train_orig)].permute(0,2,3,1).reshape(-1,21).double();cov=pixels.T@pixels/len(pixels);ev,b=torch.linalg.eigh(cov);basis=b[:,-cfg['spectral_rank']:].float()
        np.save(a.output/f'{protocol}_basis.npy',basis.numpy())
        entry={'basis_train_original_only':True,'basis_energy_fraction':float(ev[-cfg['spectral_rank']:].sum()/ev.sum()),'counts':{s:int(sum(r['split']==s and r['augmentation'] is None for r in rr)) for s in ['train','val','test']},'conditions':{}}
        for condition in cfg['controls']:
            tick=time.perf_counter();torch.manual_seed(cfg['seed']);use=train if condition!='measured_noaug' else train_orig
            kernel=np.repeat(k.mean(0,keepdims=True),21,axis=0) if condition=='achromatic_aug' else k
            physics=FFTMeasuredOperator(torch.tensor(kernel),cfg['spatial_hw'])
            with torch.no_grad():y=physics(x)
            model=SpatialSpectralPGD(kernel,basis,tuple(cfg['spatial_hw']),cfg['unrolled_stages'],cfg['prox_hidden_channels']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);train_ids=np.flatnonzero(use);history=[];grad_finite=True
            for epoch in range(cfg['epochs']):
                model.train();losses=[]
                for batch in np.array_split(np.random.default_rng(cfg['seed']+epoch).permutation(train_ids),max(1,int(np.ceil(len(train_ids)/cfg['batch_size'])))):
                    pred=model(y[batch]);target=x[batch]
                    loss=(pred['spectral_map']-target).square().mean()+(pred['global_spectrum']-target.mean((-2,-1))).square().mean()+.1*(pred['measurement_reprojection']-y[batch]).square().mean()
                    opt.zero_grad();loss.backward();grad_finite &= all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters());assert grad_finite and torch.isfinite(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            model.eval()
            with torch.no_grad():out=model(y)
            stage_features={'oracle21':x.mean((-2,-1)).numpy(),'three_band_proxy':x.mean((-2,-1)).numpy()[:,[20,10,0]],'encoded':torch.nn.functional.adaptive_avg_pool2d(y[:,None],(8,8)).flatten(1).numpy(),'n1_global':out['global_spectrum'].numpy()}
            result={'parameters':sum(p.numel() for p in model.parameters()),'train_history':history,'gradient_finite':grad_finite,'seconds':time.perf_counter()-tick,'classifiers':{},'quality':{}}
            for name,f in stage_features.items():result['classifiers'][name],failures[protocol+'_'+condition+'_'+name]=classify(f,labels,rr,use)
            for split in ['train','val','test']:
                for source in ['all','UWA','CAVE']:
                    ix=np.array([r['split']==split and r['augmentation'] is None and (source=='all' or r['source_dataset']==source) for r in rr])
                    if not ix.any():continue
                    target=x[ix];prediction=out['spectral_map'][ix];g=target.mean((-2,-1));z=out['global_spectrum'][ix];res=out['measurement_reprojection'][ix]-y[ix]
                    result['quality'][split+'_'+source]=dict(spectral_relative_l2=float((torch.linalg.vector_norm(z-g,dim=1)/torch.linalg.vector_norm(g,dim=1)).mean()),map_relative_l2=float(torch.linalg.vector_norm(prediction-target)/torch.linalg.vector_norm(target)),measurement_relative_l2=float(torch.linalg.vector_norm(res)/torch.linalg.vector_norm(y[ix])),negative_fraction=float((prediction<0).float().mean()))
            if condition=='measured_aug':
                # No fitting or selection using these perturbations: fixed sensitivity probes.
                testidx=np.flatnonzero(np.array([r['split']=='test' and r['augmentation'] is None for r in rr]));tx=x[testidx]
                weights=np.linspace(.8,1.2,21,dtype=np.float32)
                alternatives={'support1024':ks['k1024'],'nonnegative_kernel':np.maximum(k,0)/np.maximum(k,0).sum((-2,-1),keepdims=True),'spectral_weights_ramp':k*weights[:,None,None],'scale_x2':np.stack([cv2.resize(v,(16,16),interpolation=cv2.INTER_AREA)*4 for v in k])}
                sensitivity={}
                with torch.no_grad():
                    for name,alt in alternatives.items():
                        ya=FFTMeasuredOperator(torch.tensor(alt),cfg['spatial_hw'])(tx);oa=model(ya)
                        sensitivity[name]=dict(measurement_change=float(torch.linalg.vector_norm(ya-y[testidx])/torch.linalg.vector_norm(y[testidx])),global_relative_error=float((torch.linalg.vector_norm(oa['global_spectrum']-tx.mean((-2,-1)),dim=1)/torch.linalg.vector_norm(tx.mean((-2,-1)),dim=1)).mean()))
                    noise=torch.randn_like(y[testidx])*float(y[torch.tensor(train_orig)].square().mean().sqrt())*.01;no=model(y[testidx]+noise)
                    sensitivity['one_percent_train_rms_noise']={'global_relative_error':float((torch.linalg.vector_norm(no['global_spectrum']-tx.mean((-2,-1)),dim=1)/torch.linalg.vector_norm(tx.mean((-2,-1)),dim=1)).mean())}
                    zero=model(torch.zeros_like(y[testidx]));sensitivity['zero_observation']={'mean_output_norm':float(torch.linalg.vector_norm(zero['global_spectrum'],dim=1).mean())}
                result['sensitivity']=sensitivity
                np.savez_compressed(a.output/f'{protocol}_outputs.npz',sample_ids=np.array([r['sample_id'] for r in rr]),measurement=y.numpy(),spectral_map=out['spectral_map'].numpy(),global_spectrum=out['global_spectrum'].numpy(),measurement_reprojection=out['measurement_reprojection'].numpy(),target_global=x.mean((-2,-1)).numpy(),negative_fraction=out['negative_fraction'].numpy())
            torch.save(dict(state_dict=model.state_dict(),config=cfg,basis=basis,kernels=kernel),a.output/f'{protocol}_{condition}.pt')
            entry['conditions'][condition]=result;print(protocol,condition,'seconds',round(result['seconds'],1),'test',result['classifiers']['n1_global'].get('test_all'),flush=True)
        summary[protocol]=entry
        (a.output/'results.json').write_text(json.dumps(summary,indent=2))
    checks.update(seconds=time.perf_counter()-start,torch=torch.__version__,numpy=np.__version__,cpu_threads=4,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest())
    # Windows peak working set includes process allocation; this is not GPU memory.
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    memory=Memory();memory.cb=ctypes.sizeof(memory)
    if sys.platform=='win32':
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        handle=ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb):
            checks['peak_working_set_bytes']=memory.PeakWorkingSetSize
    (a.output/'runtime_checks.json').write_text(json.dumps(checks,indent=2));(a.output/'failure_predictions.json').write_text(json.dumps(failures,indent=2));print(checks)

if __name__=='__main__':main()
