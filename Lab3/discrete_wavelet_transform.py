import numpy as np

def haar1d(x):
    x=np.asarray(x,dtype=float)
    return (x[0::2]+x[1::2])/np.sqrt(2), (x[0::2]-x[1::2])/np.sqrt(2)

def inv_haar1d(a,d):
    out=np.empty(2*len(a))
    out[0::2]=(a+d)/np.sqrt(2)
    out[1::2]=(a-d)/np.sqrt(2)
    return out

def haar2d(im):
    rows,cols=im.shape
    ra=np.empty((rows,cols//2)); rd=np.empty_like(ra)
    for r in range(rows): ra[r],rd[r]=haar1d(im[r])
    LL=np.empty((rows//2,cols//2)); LH=np.empty_like(LL)
    HL=np.empty_like(LL); HH=np.empty_like(LL)
    for c in range(cols//2):
        LL[:,c],LH[:,c]=haar1d(ra[:,c])
        HL[:,c],HH[:,c]=haar1d(rd[:,c])
    return LL,LH,HL,HH

def inv_haar2d(LL,LH,HL,HH):
    h,w=LL.shape
    ra=np.empty((2*h,w)); rd=np.empty_like(ra)
    for c in range(w):
        ra[:,c]=inv_haar1d(LL[:,c],LH[:,c])
        rd[:,c]=inv_haar1d(HL[:,c],HH[:,c])
    out=np.empty((2*h,2*w))
    for r in range(2*h): out[r]=inv_haar1d(ra[r],rd[r])
    return out
