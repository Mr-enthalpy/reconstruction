"""Versioned D3 inputs from immutable D1/D2 evidence and reviewed CAVE face ROIs."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from lcd_reconstruction.datasets.controlled_views import common_scene, make_view


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ["d2", "uwa", "cave", "psf", "output"]:
        ap.add_argument("--"+name, type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old = json.loads((a.d2/"manifest.json").read_text(encoding="utf-8"))
    stored = np.load(a.d2/"scenes.npz")
    assert np.array_equal(stored["wavelengths_nm"], np.arange(450, 651, 10))
    rows, cubes, bit_scaled = [], [], []
    d1 = {r["sample_id"]: r for r in csv.DictReader((a.uwa/"dataset_manifest.csv").open())}
    inventory = {r["scene_id"]: r for r in json.loads((a.cave/"cave_inventory.json").read_text())}
    for r in old:
        if r["augmentation"] is not None:
            continue
        if r["source_dataset"] == "UWA":
            source = a.uwa/d1[r["sample_id"]]["path"]
            assert sha(source) == r["source_sha256"] == d1[r["sample_id"]]["sample_sha256"]
            raw = np.load(source)["cube"]; bits = 8
        else:
            source = a.cave/inventory[r["scene_id"]]["cube_path"]
            raw = np.load(source)["cube"]; bits = 16
        x, factor = common_scene(raw, bits)
        assert np.allclose(x, stored["cube"][r["index"]], rtol=1e-6, atol=1e-7)
        record = dict(r, index=len(rows), dataset_id=r["source_dataset"],
                      source_scene_or_acquisition_id=r["parent_id"],
                      source_group_id=("UWA_subject_"+r["subject_id"] if bits == 8 else "CAVE_"+r["parent_id"]),
                      class_label=r["label"], label_definition="recognizable human face in candidate region, including photographs",
                      source_radiometric_type=r["source_type"],
                      presentation_type="physical_person" if r["label"] == "face" else "not_applicable",
                      quality_flags=["D1 accepted; residual registration/quantization limits"] if bits == 8 else ["approximate_reflectance; source domain differs"],
                      input_file_sha256=sha(source))
        rows.append(record); cubes.append(x); bit_scaled.append(x/factor)
    # Coordinates reviewed on the actual 512x512 650/550/450 composite.
    specs = [
        ("face", "physical_person", [112, 64, 400, 400]),
        ("photo_and_face", "physical_person", [272, 176, 240, 272]),
        ("photo_and_face", "printed_photo", [16, 224, 176, 224]),
    ]
    tiles = []
    for scene, presentation, crop in specs:
        source = a.cave/(scene+".npz")
        raw = np.load(source); assert raw["cube"].dtype == np.uint16
        assert np.array_equal(raw["wavelengths_nm"], np.arange(450, 651, 10))
        x, y, w, h = crop
        assert 0 <= x < x+w <= 512 and 0 <= y < y+h <= 512
        cube = raw["cube"][:, y:y+h, x:x+w]
        assert cube.shape == (21, h, w) and np.isfinite(cube).all()
        xx, factor = common_scene(cube, 16)
        record = dict(index=len(rows), sample_id="CAVE_"+scene+"_"+presentation, parent_id=scene,
                      scene_id=scene, subject_id=None, source_dataset="CAVE", dataset_id="CAVE",
                      source_scene_or_acquisition_id=scene, source_group_id="CAVE_face_family_unverified_independence",
                      split="train", label="face", class_label="face",
                      label_definition="recognizable human face in candidate region, including photographs",
                      presentation_type=presentation, source_type="approximate_reflectance",
                      source_radiometric_type="approximate_reflectance", bit_depth=16, crop_xywh=crop,
                      wavelengths_nm=list(range(450,651,10)), augmentation=None,
                      normalization_factor=factor, normalization_policy="D2 dimensionless grand mean 0.2",
                      source_sha256=inventory[scene]["source_sha256"], input_file_sha256=sha(source),
                      label_review_status="Agent actual-image content review; not independent human adjudication",
                      quality_flags=["only two source scenes; grouped together; train-only", "aspect resized to 32x32", "no physical radiometry"])
        rows.append(record); cubes.append(xx); bit_scaled.append(xx/factor)
        rgb=np.moveaxis(cube[[20,10,0]],0,-1).astype(float)
        rgb=np.clip(rgb/max(np.quantile(rgb,.99),1),0,1)**.6
        tile=Image.new("RGB",(256,282),"white")
        tile.paste(Image.fromarray((rgb*255).astype("uint8")).resize((256,256)),(0,26))
        ImageDraw.Draw(tile).text((2,2),scene+" "+presentation,fill="black");tiles.append(tile)
    original_count=len(rows)
    for i in range(original_count):
        r=rows[i]
        if r["split"]!="train":
            continue
        seed=int(hashlib.sha256(("D3_"+r["sample_id"]).encode()).hexdigest()[:8],16)
        # Preserve the existing strict whole-ROI guard; deterministically reject
        # a sampled transform that would crop corners rather than weaken it.
        for attempt in range(100):
            try:
                v,params=make_view(cubes[i],seed+attempt)
                break
            except ValueError as exc:
                if str(exc)!="transform crops ROI semantics":
                    raise
        else:
            raise ValueError("no whole-ROI augmentation within 100 seeded attempts")
        params.update(initial_seed=seed,rejected_transform_count=attempt)
        replay,_=make_view(cubes[i],params["seed"])
        assert np.array_equal(v,replay)
        rows.append(dict(r,index=len(rows),sample_id=r["sample_id"]+"_aug",augmentation=params))
        cubes.append(v);bit_scaled.append(v/r["normalization_factor"])
    groups={}
    for r in rows:
        assert groups.setdefault(r["source_group_id"],r["split"])==r["split"]
        assert r["augmentation"] is None or r["split"]=="train"
    psf_audit=json.loads((a.d2/"psf_audit.json").read_text())
    for r in psf_audit:
        assert sha(a.psf/f'roi_extent_review/roi1024_{r["wavelength_nm"]}.npz')==r["source_sha256"]
    kernels=np.load(a.d2/"kernels.npz")
    np.savez_compressed(a.output/"scenes.npz",cube=np.stack(cubes),bit_scaled=np.stack(bit_scaled),
                        global_spectrum=np.stack(cubes).mean((-2,-1)),wavelengths_nm=np.arange(450,651,10))
    np.savez_compressed(a.output/"kernels.npz",**{k:kernels[k] for k in kernels.files})
    (a.output/"manifest.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (a.output/"splits.json").write_text(json.dumps({s:[r["sample_id"] for r in rows if r["split"]==s] for s in ["train","val","test"]},indent=2))
    with (a.output/"dataset_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    preview=Image.new("RGB",(768,282),"white")
    for i,t in enumerate(tiles):preview.paste(t,(256*i,0))
    preview.save(a.output/"private_face_roi_review.png")
    counts={s:{d:{label:sum(r["split"]==s and r["dataset_id"]==d and r["label"]==label and r["augmentation"] is None for r in rows) for label in ["face","non_face"]} for d in ["UWA","CAVE"]} for s in ["train","val","test"]}
    report=dict(original_rois=original_count,augmented_training_views=len(rows)-original_count,counts=counts,
                cave_face_source_scenes=2,cave_face_conservative_groups=1,cave_face_rois=3,
                group_isolation=True,replay=True,psf_source_verified=True,
                d2_scenes_sha256=sha(a.d2/"scenes.npz"),d2_manifest_sha256=sha(a.d2/"manifest.json"),
                kernels_sha256=sha(a.d2/"kernels.npz"),d1_manifest_sha256=sha(a.uwa/"dataset_manifest.csv"),
                d1_splits_sha256=sha(a.uwa/"splits.json"))
    (a.output/"validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
