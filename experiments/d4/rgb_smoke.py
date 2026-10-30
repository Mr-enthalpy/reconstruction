"""Local mature-detector interface check; Oracle proxy is NOT recovered UDC RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from lcd_reconstruction.rgb_frontend import recover_rgb, propose_faces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cave", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for scene in ("face", "photo_and_face", "balloons"):
        source = a.cave/f"{scene}.npz"
        with np.load(source) as z:
            # Fixed 650/550/450 proxy; no colour-response claim and no truth from N1.
            rgb = np.moveaxis(z["cube"][[20, 10, 0]], 0, -1)
        raw = np.empty(rgb.shape[:2], np.uint16)
        raw[::2, ::2] = rgb[::2, ::2, 0]
        raw[::2, 1::2] = rgb[::2, 1::2, 1]
        raw[1::2, ::2] = rgb[1::2, ::2, 1]
        raw[1::2, 1::2] = rgb[1::2, 1::2, 2]
        result = recover_rgb(raw, cfa="RGGB", black_level=0, white_level=65535,
                             observation_id=f"ORACLE_PROXY_SYNTHETIC_MOSAIC_{scene}")
        result = propose_faces(result, a.model)
        preview = np.ascontiguousarray(np.round(np.clip(result["rgb_image"], 0, 1)*255).astype(np.uint8)[..., ::-1])
        for box in result["candidate_boxes"]:
            x, y, w, h = np.round(box).astype(int)
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imwrite(str(a.output/f"private_{scene}.png"), preview)
        results.append(dict(scene=scene, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            boxes_xywh=result["candidate_boxes"].tolist(),
                            scores=result["candidate_scores"].tolist(), metadata=result["metadata"]))
    (a.output/"results.json").write_text(json.dumps(dict(
        opencv_version=cv2.__version__, results=results, udc_degradation_tested=False,
        detector_accuracy_validated=False, rgb_type="Oracle three-band proxy through synthetic Bayer",
        n1_shared_raw_test=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
