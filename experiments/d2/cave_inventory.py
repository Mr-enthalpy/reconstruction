"""Read the official complete CAVE ZIP; preserve every format exception."""
from pathlib import Path
import argparse,zipfile,re,json,hashlib,csv
import cv2,numpy as np
from PIL import Image,ImageDraw

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    archive_sha=hashlib.file_digest(a.archive.open('rb'),'sha256').hexdigest();rows=[];tiles=[];band_rows=[]
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None
        scenes=sorted({n.split('/')[0] for n in z.namelist() if re.search(r'_\d\d\.png$',n,re.I)})
        for scene in scenes:
            bands={};errors=[];content_hash=hashlib.sha256()
            for name in z.namelist():
                m=re.search(r'_(\d\d)\.png$',name,re.I)
                if not m or not name.startswith(scene+'/'):continue
                b=int(m[1]);raw=z.read(name);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
                assert b not in bands;bands[b]=im
                band_rows.append(dict(scene=scene,member=name,band=b,wavelength_nm=390+10*b,shape=str(im.shape),dtype=str(im.dtype),sha256=hashlib.sha256(raw).hexdigest(),min=int(im.min()),max=int(im.max()),finite=bool(np.isfinite(im).all())))
                if im.shape!=(512,512) or im.dtype!=np.uint16:errors.append(dict(member=name,shape=list(im.shape),dtype=str(im.dtype)))
            assert sorted(bands)==list(range(1,32))
            for b in sorted(bands):content_hash.update(bands[b].tobytes())
            sid=scene.removesuffix('_ms');row=dict(scene_id=sid,parent_scene_id=sid,source_dataset='CAVE',source_type='approximate_reflectance',bit_depth=16 if not errors else 'format_exception',crop_xywh=[0,0,512,512],wavelengths_nm=list(range(450,651,10)),source_url='https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip',source_sha256=archive_sha,content_sha256=content_hash.hexdigest(),normalization_policy='raw preserved; simulation normalization separate',label_review_status='rejected_format' if errors else 'pending',format_errors=errors)
            if not errors:
                cube=np.stack([bands[b] for b in range(6,27)]);file=a.output/f'{sid}.npz';np.savez_compressed(file,cube=cube,wavelengths_nm=np.arange(450,651,10));row.update(cube_path=file.name,shape=list(cube.shape),min=int(cube.min()),max=int(cube.max()),nonfinite=0)
                rgb=np.moveaxis(cube[[20,10,0]],0,-1).astype(float);rgb=np.clip(rgb/max(np.quantile(rgb,.99),1),0,1)**.6
                tile=Image.new('RGB',(256,282),'white');tile.paste(Image.fromarray((rgb*255).astype('uint8')).resize((256,256)),(0,26));ImageDraw.Draw(tile).text((4,4),sid,fill='black');tiles.append(tile)
            rows.append(row)
    (a.output/'cave_inventory.json').write_text(json.dumps(rows,indent=2))
    with (a.output/'band_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(band_rows[0]));w.writeheader();w.writerows(band_rows)
    for s in range(0,len(tiles),8):
        montage=Image.new('RGB',(1024,564),'white')
        for j,t in enumerate(tiles[s:s+8]):montage.paste(t,((j%4)*256,(j//4)*282))
        montage.save(a.output/f'review_{s//8}.png')
    duplicates={r['scene_id']:[v['scene_id'] for v in rows if v['content_sha256']==r['content_sha256'] and v['scene_id']!=r['scene_id']] for r in rows}
    (a.output/'integrity.json').write_text(json.dumps(dict(archive_sha256=archive_sha,scenes=len(rows),decoded_bands=len(band_rows),format_rejected=[r['scene_id'] for r in rows if r['format_errors']],exact_duplicate_scenes={k:v for k,v in duplicates.items() if v}),indent=2));print((a.output/'integrity.json').read_text())

if __name__=='__main__':main()
