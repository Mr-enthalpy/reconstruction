"""One shared affine transform for every band; entire ROI stays on the canvas."""
import cv2
import numpy as np


def make_view(cube,seed):
    rng=np.random.default_rng(seed);h,w=cube.shape[-2:]
    scale=float(rng.uniform(.88,.94));angle=float(rng.uniform(-3,3))
    flip=bool(rng.random()<.5);shift=rng.uniform(-.5,.5,2)
    matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,scale)
    matrix[:,2]+=shift
    corners=np.array([[0,0,1],[w-1,0,1],[0,h-1,1],[w-1,h-1,1]])@matrix.T
    if (corners<0).any() or (corners[:,0]>w-1).any() or (corners[:,1]>h-1).any():
        raise ValueError('transform crops ROI semantics')
    src=cube[:,:,::-1] if flip else cube
    view=np.stack([cv2.warpAffine(a,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for a in src])
    brightness=float(rng.uniform(.95,1.05));view*=brightness
    return view,dict(seed=int(seed),affine=matrix.tolist(),horizontal_flip=flip,brightness=brightness,corners_inside=True)


def common_scene(cube,bit_depth,hw=(32,32)):
    a=np.asarray(cube,dtype=np.float32)/(2**bit_depth-1)
    a=np.stack([cv2.resize(b,hw[::-1],interpolation=cv2.INTER_AREA) for b in a])
    if a.shape[0]!=21 or not np.isfinite(a).all() or a.mean()<=0:
        raise ValueError('invalid scene')
    # Explicit energy-matched pseudo-scene, never a claim of shared physical units.
    factor=float(.2/a.mean())
    return a*factor,factor
