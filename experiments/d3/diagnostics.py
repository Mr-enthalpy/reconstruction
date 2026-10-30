"""Local evidence and aggregate plots; no face imagery is written to public reports."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lcd_reconstruction.unrolling.full_spectral import SinglePlanePhysics, FullSpectralPGD


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared",type=Path,required=True);ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    x=np.load(a.prepared/"scenes.npz")["cube"];rows=json.loads((a.prepared/"manifest.json").read_text(encoding="utf-8"))
    results=json.loads((a.run/"results.json").read_text())
    k=np.load(a.prepared/"kernels.npz")["k512"]
    op=SinglePlanePhysics(torch.tensor(k,dtype=torch.float64),(32,32))
    rng=np.random.default_rng(82);z=torch.tensor(rng.normal(size=(1,21,32,32)),dtype=torch.float64)
    def matvec(v):
        yy=torch.tensor(v.reshape(1,1,32,32),dtype=torch.float64)
        return op(op.adjoint(yy)).numpy().ravel()
    rhs=op(z).numpy().ravel()
    solution,info=cg(LinearOperator((1024,1024),matvec=matvec,dtype=np.float64),rhs,
                     rtol=1e-9,atol=0,maxiter=500)
    d=z-op.adjoint(torch.tensor(solution.reshape(1,1,32,32)))
    # Two nonnegative cubes around a constant scene; this is an identifiability
    # demonstration, not a realistic scene distribution or a new training sample.
    delta=.1*d/d.abs().max();base=torch.ones_like(delta)
    null=dict(cg_info=int(info),measurement_relative_change=float(torch.linalg.vector_norm(op(base+delta)-op(base))/torch.linalg.vector_norm(op(base))),
              cube_relative_change=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(base)),
              both_nonnegative=bool((base+delta>=0).all()),measurement_dimensions=1024,scene_dimensions=21504)
    # Actual measured-kernel directional derivative check.
    target=torch.tensor(rng.normal(size=(1,1,32,32)),dtype=torch.float64)
    grad=op.adjoint(op(z)-target)
    loss=lambda v:.5*(op(v)-target).square().sum()
    direction=torch.tensor(rng.normal(size=z.shape),dtype=torch.float64)
    numerical=(loss(z+1e-6*direction)-loss(z-1e-6*direction))/2e-6
    analytic=(grad*direction).sum()
    null["directional_absolute_error"]=float(abs(numerical-analytic))
    null["directional_relative_error"]=float(abs(numerical-analytic)/analytic.abs().clamp_min(1e-10))
    state=torch.load(a.run/"measured_aug.pt",weights_only=False)
    model=FullSpectralPGD(state["kernels"]).eval();model.load_state_dict(state["state_dict"]);model.requires_grad_(False)
    y=model.physics(torch.tensor(x[:1]))
    with torch.no_grad():
        for _ in range(5):model(y)
        tick=time.perf_counter()
        for _ in range(50):model(y)
    null["inference_seconds_per_scene_cpu4"]=(time.perf_counter()-tick)/50
    (a.output/"numerical_diagnostics.json").write_text(json.dumps(null,indent=2))
    # Aggregate-only publication figures.
    nm=np.arange(450,651,10);fig,ax=plt.subplots(figsize=(8,4))
    for name,r in results["conditions"].items():
        ax.plot(nm,r["reconstruction"]["test_all"]["band_relative_l2"],label=name)
    ax.set(xlabel="Nominal wavelength (nm)",ylabel="Test relative band L2");ax.legend();fig.tight_layout()
    fig.savefig(a.output/"aggregate_band_errors.png",dpi=150);plt.close(fig)
    names=["oracle","reconstruction","frozen_head_transfer","oracle_proxy","oracle_intensity","encoded_plane"]
    fig,ax=plt.subplots(figsize=(9,4));t=np.arange(len(names))
    for j,c in enumerate(["measured_aug","achromatic_aug","measured_noaug"]):
        ax.bar(t+(j-1)*.25,[results["conditions"][c]["classification"][n]["test_all"]["balanced_accuracy"] for n in names],.25,label=c)
    ax.set(xticks=t,xticklabels=["Oracle","Reconstruction","Frozen head","3-band proxy","Intensity","Encoded"],ylim=(0,1.1),ylabel="Test balanced accuracy")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/"aggregate_classification.png",dpi=150);plt.close(fig)
    # Private example uses a held-out non-face scene. Outputs still remain local.
    outputs=np.load(a.run/"measured_aug_outputs.npz")
    i=next(i for i,r in enumerate(rows) if r["dataset_id"]=="CAVE" and r["split"]=="test" and r["augmentation"] is None)
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for j,b in enumerate([0,10,20]):
        vmin=min(x[i,b].min(),outputs["spectral_map"][i,b].min());vmax=max(x[i,b].max(),outputs["spectral_map"][i,b].max())
        for row,img in enumerate([x[i,b],outputs["spectral_map"][i,b]]):
            axes[row,j].imshow(img,vmin=vmin,vmax=vmax,cmap="viridis");axes[row,j].set_title(("Target " if row==0 else "N1 ")+str(nm[b])+" nm");axes[row,j].axis("off")
    fig.tight_layout();fig.savefig(a.output/"private_scene_example.png",dpi=150);plt.close(fig)
    checks=dict(original_count=sum(r["augmentation"] is None for r in rows),
                finite=bool(np.isfinite(x).all()),shape=list(x.shape),
                wavelengths_nm=nm.tolist(),supervision_mean_error=float(np.max(np.abs(x.mean((-2,-1))-np.load(a.prepared/"scenes.npz")["global_spectrum"]))),
                rejected_augmentation_draws=sum((r["augmentation"] or {}).get("rejected_transform_count",0) for r in rows))
    (a.output/"final_data_checks.json").write_text(json.dumps(checks,indent=2))
    print(null);print(checks)


if __name__=="__main__":
    main()
