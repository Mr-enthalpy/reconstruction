"""Fixed-budget D3: reconstruct scenes without labels, then freeze and classify."""
import argparse, ctypes, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics, FullSpectralPGD, scene_loss


def digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit_head(features,labels,train):
    """Balanced logistic regression; reusable fitted weights enable frozen transfer."""
    z=np.asarray(features,dtype=np.float64);mu=z[train].mean(0);sd=z[train].std(0)
    sd[sd<1e-9]=1
    x=np.column_stack([np.ones(len(z)),(z-mu)/sd]);xt=x[train];yt=labels[train]
    count=np.bincount(yt,minlength=2)
    if not (count>0).all():raise ValueError("both training classes required")
    weights=1/count[yt];weights/=weights.sum()
    def objective(w):
        v=xt@w;reg=w.copy();reg[0]=0
        return (float((weights*(np.logaddexp(0,v)-yt*v)).sum()+.005*reg@reg),
                xt.T@(weights*(expit(v)-yt))+.01*reg)
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                 options={"maxiter":2000})
    if not fit.success:raise RuntimeError(fit.message)
    return dict(mu=mu,sd=sd,weights=fit.x)


def predict(head,features):
    x=np.column_stack([np.ones(len(features)),(features-head["mu"])/head["sd"]])
    return (expit(x@head["weights"])>=.5).astype(int)


def evaluate(pred,labels,records):
    results={}
    for split in ["train","val","test"]:
        for domain in ["all","UWA","CAVE"]:
            ids=[i for i,r in enumerate(records) if r["split"]==split and r["augmentation"] is None and (domain=="all" or r["dataset_id"]==domain)]
            if not ids:continue
            cm=np.zeros((2,2),int)
            for i in ids:cm[labels[i],pred[i]]+=1
            n=cm.sum(1);recall=np.divide(cm.diagonal(),n,out=np.zeros(2,float),where=n>0)
            den=cm.sum(0)+n;f1=np.divide(2*cm.diagonal(),den,out=np.zeros(2,float),where=den>0)
            results[split+"_"+domain]=dict(n=len(ids),confusion_matrix=cm.tolist(),
                accuracy=float(cm.trace()/len(ids)),balanced_accuracy=float(recall.mean()) if (n>0).all() else None,
                macro_f1=float(f1.mean()) if (n>0).all() else None,recall=recall.tolist(),support=n.tolist())
    return results


def features(x,kind):
    if kind=="global":return x.mean((-2,-1)).numpy()
    if kind=="proxy":x=x[:,[20,10,0]]
    if kind=="intensity":x=x.mean(1,keepdim=True)
    return F.adaptive_avg_pool2d(x,(4,4)).flatten(1).numpy()


def source_features(x):
    return np.array([[a.mean(),a.std(),*np.quantile(a,[.01,.1,.5,.9,.99]),
                      (a==0).mean(),np.abs(np.diff(a.mean(0),axis=0)).mean(),
                      np.abs(np.diff(a.mean(0),axis=1)).mean()] for a in x])


def quality(estimate,target,reprojection,y):
    err=estimate-target
    norm=lambda a:torch.linalg.vector_norm(a)
    safe=lambda a:a.clamp_min(1e-10)
    valid=torch.linalg.vector_norm(target,dim=1)>1e-3
    cos=F.cosine_similarity(estimate,target,dim=1,eps=1e-8).clamp(-1,1)
    sam=torch.acos(cos[valid])*180/np.pi
    grad_err=sum(torch.diff(err,dim=d).square().sum() for d in [-1,-2])
    grad_ref=sum(torch.diff(target,dim=d).square().sum() for d in [-1,-2])
    band=torch.sqrt(err.square().sum((0,2,3))/safe(target.square().sum((0,2,3))))
    ef=torch.fft.fft2(err);tf=torch.fft.fft2(target)
    fy=torch.fft.fftfreq(target.shape[-2])[:,None];fx=torch.fft.fftfreq(target.shape[-1])[None,:]
    high=(fx.square()+fy.square()).sqrt()>=.25
    return dict(map_relative_l2=float(norm(err)/safe(norm(target))),
        spatial_gradient_relative_l2=float(torch.sqrt(grad_err/safe(grad_ref))),
        spectral_angle_deg=float(sam.mean()) if sam.numel() else None,
        valid_spectral_pixel_fraction=float(valid.float().mean()),
        global_spectrum_relative_l2=float((norm_batch(estimate.mean((-2,-1))-target.mean((-2,-1)))/safe(norm_batch(target.mean((-2,-1))))).mean()),
        measurement_relative_l2=float(norm(reprojection-y)/safe(norm(y))),
        negative_fraction=float((estimate<0).float().mean()),band_relative_l2=band.tolist(),
        high_frequency_relative_l2=float(norm(ef[...,high])/safe(norm(tf[...,high]))),
        low_frequency_relative_l2=float(norm(ef[...,~high])/safe(norm(tf[...,~high]))))


def norm_batch(x):
    return torch.linalg.vector_norm(x,dim=1)


def train_n1(x,train,kernel,cfg):
    """No class/domain labels enter this function or objective."""
    torch.manual_seed(cfg["seed"])
    physics=SinglePlanePhysics(torch.tensor(kernel),tuple(cfg["hw"]))
    with torch.no_grad():y=physics(x)
    scale=y[train].square().mean().sqrt()
    model=FullSpectralPGD(kernel,tuple(cfg["hw"]),cfg["stages"],cfg["hidden"])
    opt=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    ids=np.flatnonzero(train);history=[]
    for epoch in range(cfg["epochs"]):
        order=np.random.default_rng(cfg["seed"]+epoch).permutation(ids);losses=[]
        model.train()
        for start in range(0,len(ids),cfg["batch"]):
            batch=order[start:start+cfg["batch"]]
            out=model(y[batch]);loss=scene_loss(out,x[batch],scale)
            opt.zero_grad();loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite training")
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval().requires_grad_(False)
    with torch.no_grad():out=model(y)
    return model,physics,y,out,history,float(scale)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());torch.set_num_threads(4);t0=time.perf_counter()
    if cfg["noise_train_rms_fraction"] != 0:
        raise ValueError("this fixed pilot trains noiselessly; noise is an evaluation sensitivity")
    data=np.load(a.prepared/"scenes.npz");x=torch.tensor(data["cube"]);ks=np.load(a.prepared/"kernels.npz")
    k=ks["k512"];records=json.loads((a.prepared/"manifest.json").read_text(encoding="utf-8"))
    original=np.array([r["augmentation"] is None for r in records])
    train=np.array([r["split"]=="train" for r in records]);labels=np.array([r["class_label"]=="face" for r in records],int)
    result={"config":cfg,"source_diagnostic":{},"conditions":{}};predictions={}
    # Class-conditional source check avoids interpreting face-vs-object as source detection.
    ni=[i for i,r in enumerate(records) if r["class_label"]=="non_face"];nr=[records[i] for i in ni]
    sy=np.array([r["dataset_id"]=="CAVE" for r in nr],int)
    for name,xx,aug in [("bit_scaled_noaug",data["bit_scaled"],False),
                        ("equal_energy_noaug",data["cube"],False),("equal_energy_aug",data["cube"],True)]:
        tr=np.array([r["split"]=="train" and (aug or r["augmentation"] is None) for r in nr])
        sf=source_features(xx[ni]);head=fit_head(sf,sy,tr)
        result["source_diagnostic"][name]=dict(class_order=["UWA","CAVE"],metrics=evaluate(predict(head,sf),sy,nr))
    for condition in cfg["conditions"]:
        tick=time.perf_counter();use=train if condition!="measured_noaug" else train&original
        kernel=np.repeat(k.mean(0,keepdims=True),21,axis=0) if condition=="achromatic_aug" else k
        model,physics,y,out,history,scale=train_n1(x,use,kernel,cfg)
        state_hash=digest(model);rec=out["spectral_map"].detach()
        entry=dict(parameters=sum(p.numel() for p in model.parameters()),seconds_training_and_inference=time.perf_counter()-tick,
                   history=history,finite_gradients=True,frozen_n1_sha256=state_hash,
                   measurement_train_rms=scale,reconstruction={},classification={},sensitivity={})
        for split in ["train","val","test"]:
            for domain in ["all","UWA","CAVE"]:
                ids=[i for i,r in enumerate(records) if r["split"]==split and r["augmentation"] is None and (domain=="all" or r["dataset_id"]==domain)]
                if ids:entry["reconstruction"][split+"_"+domain]=quality(rec[ids],x[ids],out["measurement_reprojection"][ids],y[ids])
        oracle=features(x,"cube");recovered=features(rec,"cube")
        oh=fit_head(oracle,labels,use);rh=fit_head(recovered,labels,use)
        trials={"oracle":(oh,oracle),"reconstruction":(rh,recovered),"frozen_head_transfer":(oh,recovered)}
        for kind in ["global","proxy","intensity"]:
            f=features(x,kind);trials["oracle_"+kind]=(fit_head(f,labels,use),f)
        f=features(rec,"global");trials["reconstruction_global"]=(fit_head(f,labels,use),f)
        f=features(y,"cube");trials["encoded_plane"]=(fit_head(f,labels,use),f)
        # UWA-only training diagnostic, evaluated on the same untouched test originals.
        uwa_use=use & np.array([r["dataset_id"]=="UWA" for r in records])
        trials["oracle_uwa_only_train"]=(fit_head(oracle,labels,uwa_use),oracle)
        for name,(head,f) in trials.items():
            pred=predict(head,f);entry["classification"][name]=evaluate(pred,labels,records)
            predictions[condition+"_"+name]=[dict(sample_id=r["sample_id"],source_group=r["source_group_id"],split=r["split"],truth=int(labels[i]),prediction=int(pred[i])) for i,r in enumerate(records) if original[i]]
            np.savez_compressed(a.output/(condition+"_"+name+"_head.npz"),**head)
        assert digest(model)==state_hash and all(not p.requires_grad for p in model.parameters())
        entry["n1_unchanged_by_classification"]=True
        ti=np.flatnonzero(np.array([r["split"]=="test" for r in records])&original)
        if condition=="measured_aug":
            gen=torch.Generator().manual_seed(cfg["seed"]+919)
            with torch.no_grad():
                for noise in cfg["noise_sensitivity_fractions"]:
                    yn=y[ti]+torch.randn(y[ti].shape,generator=gen)*scale*noise
                    on=model(yn);entry["sensitivity"][f"gaussian_{noise}"]=quality(on["spectral_map"],x[ti],on["measurement_reprojection"],yn)
                alternatives={"support1024":ks["k1024"],
                    "spectral_weights_ramp":k*np.linspace(.8,1.2,21,dtype=np.float32)[:,None,None],
                    "nonnegative_kernels":np.maximum(k,0)/np.maximum(k,0).sum((-2,-1),keepdims=True)}
                for name,alt in alternatives.items():
                    yy=SinglePlanePhysics(torch.tensor(alt),tuple(cfg["hw"]))(x[ti]);oo=model(yy)
                    entry["sensitivity"][name]=quality(oo["spectral_map"],x[ti],oo["measurement_reprojection"],yy)
                # Hypothetical RGGB spectral responses; explicitly not sensor calibration.
                nm=np.arange(450,651,10)
                responses=np.stack([np.exp(-.5*((nm-c)/50)**2) for c in [650,550,450]])
                responses/=responses.mean(0,keepdims=True)
                planes=[SinglePlanePhysics(torch.tensor(k*rr[:,None,None],dtype=torch.float32),tuple(cfg["hw"]))(x[ti]) for rr in responses]
                mosaic=planes[1].clone();mosaic[:,:,0::2,0::2]=planes[0][:,:,0::2,0::2];mosaic[:,:,1::2,1::2]=planes[2][:,:,1::2,1::2]
                oo=model(mosaic);entry["sensitivity"]["hypothetical_rggb_mismatch"]=quality(oo["spectral_map"],x[ti],oo["measurement_reprojection"],mosaic)
                zero=model(torch.zeros_like(y[ti]))
                entry["sensitivity"]["zero_measurement"]={"mean_spectrum_norm":float(norm_batch(zero["global_spectrum"]).mean())}
        torch.save(dict(state_dict=model.state_dict(),kernels=kernel,config=cfg),a.output/(condition+".pt"))
        np.savez_compressed(a.output/(condition+"_outputs.npz"),spectral_map=rec.numpy(),
                            global_spectrum=out["global_spectrum"].numpy(),measurement=y.numpy(),
                            measurement_reprojection=out["measurement_reprojection"].numpy(),
                            sample_ids=np.array([r["sample_id"] for r in records]))
        # Local-only per-case reconstruction failures, not public individual face records.
        errors=torch.linalg.vector_norm((rec-x).flatten(1),dim=1)/torch.linalg.vector_norm(x.flatten(1),dim=1)
        predictions[condition+"_reconstruction_errors"]=[dict(sample_id=r["sample_id"],split=r["split"],relative_l2=float(errors[i])) for i,r in enumerate(records) if original[i]]
        result["conditions"][condition]=entry
        (a.output/"results.json").write_text(json.dumps(result,indent=2))
        print(condition,entry["reconstruction"]["test_all"],flush=True)
        print("C1", {n:v["test_all"]["balanced_accuracy"] for n,v in entry["classification"].items()},flush=True)
    result["runtime"]=dict(seconds=time.perf_counter()-t0,torch=torch.__version__,numpy=np.__version__,
                          config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest())
    if sys.platform=="win32":
        class Memory(ctypes.Structure):
            _fields_=[("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)]+[(s,ctypes.c_size_t) for s in ["peak","working","qpp","qp","qnp","qn","page","peakpage"]]
        m=Memory();m.cb=ctypes.sizeof(m);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),ctypes.byref(m),m.cb):
            result["runtime"]["peak_working_set_bytes"]=m.peak
    (a.output/"results.json").write_text(json.dumps(result,indent=2))
    (a.output/"private_predictions.json").write_text(json.dumps(predictions,indent=2))
    print(result["runtime"])


if __name__=="__main__":
    main()
