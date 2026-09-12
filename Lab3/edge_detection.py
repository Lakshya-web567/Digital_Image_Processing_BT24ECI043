import numpy as np

def first_order_edges(im):
    p=np.pad(im,((1,1),(1,1)),mode="edge")
    forward=np.hypot(p[1:-1,2:]-p[1:-1,1:-1],p[2:,1:-1]-p[1:-1,1:-1])
    backward=np.hypot(p[1:-1,1:-1]-p[1:-1,:-2],p[1:-1,1:-1]-p[:-2,1:-1])
    central=np.hypot((p[1:-1,2:]-p[1:-1,:-2])/2,(p[2:,1:-1]-p[:-2,1:-1])/2)
    return forward,backward,central

def gaussian_kernel(size,sigma):
    r=size//2; y,x=np.mgrid[-r:r+1,-r:r+1]
    k=np.exp(-(x*x+y*y)/(2*sigma*sigma)); return k/k.sum()

def log_kernel(size=9,sigma=1.4):
    r=size//2; y,x=np.mgrid[-r:r+1,-r:r+1]; s2=sigma*sigma
    k=((x*x+y*y-2*s2)/(s2*s2))*np.exp(-(x*x+y*y)/(2*s2))
    return k-k.mean()

def conv2(im,k):
    kh,kw=k.shape; ph,pw=kh//2,kw//2
    p=np.pad(im,((ph,ph),(pw,pw)),mode="edge")
    windows=np.lib.stride_tricks.sliding_window_view(p,(kh,kw))
    return np.einsum("ijkl,kl->ij",windows,k,optimize=True)

def zero_cross(response,threshold):
    out=np.zeros_like(response,dtype=np.uint8)
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        shifted=np.roll(np.roll(response,dy,0),dx,1)
        out|=(((response*shifted)<0)&(np.abs(response-shifted)>=threshold)).astype(np.uint8)*255
    out[:2,:]=out[-2:,:]=out[:,:2]=out[:,-2:]=0
    return out
