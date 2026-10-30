"""Offline linear algebra prototype; not a calibrated camera simulator."""
import numpy as np
from scipy.signal import fftconvolve
class MeasuredPSFOperator:
 def __init__(self,kernels,weights):
  self.k=np.asarray(kernels,dtype=float);self.w=np.asarray(weights,dtype=float)
  if self.k.ndim!=3 or self.k.shape[0]!=21 or self.w.shape!=(21,):raise ValueError("21 kernels/explicit weights required")
  if not np.isfinite(self.k).all() or not np.isfinite(self.w).all() or (self.w<0).any():raise ValueError("finite kernels and nonnegative weights required")
 def forward(self,x):
  x=np.asarray(x,dtype=float)
  if x.ndim!=3 or x.shape[0]!=21 or not np.isfinite(x).all():raise ValueError("x must be finite [21,H,W]")
  h,w=x.shape[1:];oy,ox=np.array(self.k.shape[1:])//2
  y=np.zeros((h,w))
  for a,k,weight in zip(x,self.k,self.w):y+=weight*fftconvolve(a,k,mode="full")[oy:oy+h,ox:ox+w]
  return y
 def adjoint(self,y):
  y=np.asarray(y,dtype=float)
  if y.ndim!=2 or not np.isfinite(y).all():raise ValueError("y must be finite [H,W]")
  h,w=y.shape;kh,kw=self.k.shape[1:];oy,ox=kh//2,kw//2
  padded=np.zeros((h+kh-1,w+kw-1));padded[oy:oy+h,ox:ox+w]=y
  return np.stack([weight*fftconvolve(padded,k[::-1,::-1],mode="valid") for k,weight in zip(self.k,self.w)])
def spectral_target(x):
 x=np.asarray(x,dtype=float)
 if x.ndim!=4 or x.shape[1]!=21 or not np.isfinite(x).all():raise ValueError("expected finite [batch,21,H,W]")
 return x.mean(axis=(-2,-1))
