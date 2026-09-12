from pathlib import Path
import numpy as np
from PIL import Image
from discrete_wavelet_transform import haar2d,inv_haar2d
from shannon_fano_coding import shannon_fano_codes,entropy
from huffman_coding import huffman_codes
from edge_detection import first_order_edges,gaussian_kernel,log_kernel,conv2,zero_cross

BASE=Path(__file__).parent
INPUT=BASE/"input"/"nature_processing_600x400.jpg"

rgb=np.asarray(Image.open(INPUT).convert("RGB"),dtype=float)
gray=0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]

# DWT/IDWT
LL,LH,HL,HH=haar2d(gray)
recon=inv_haar2d(LL,LH,HL,HH)
print("DWT/IDWT MSE:",np.mean((gray-recon)**2))

# Shannon-Fano / Huffman
g=np.uint8(np.clip(np.rint(gray),0,255))
counts=np.bincount(g.ravel(),minlength=256)
symbols=np.nonzero(counts)[0]
probs=counts[symbols]/counts.sum()
pairs=list(zip(symbols,probs))
sf=shannon_fano_codes(pairs)
hf=huffman_codes(pairs)
H=entropy(probs)
sf_avg=sum(p*len(sf[s]) for s,p in pairs)
hf_avg=sum(p*len(hf[s]) for s,p in pairs)
print("Entropy:",H,"bits/pixel")
print("Shannon-Fano:",sf_avg,"bits/symbol,",100*H/sf_avg,"% efficiency")
print("Huffman:",hf_avg,"bits/symbol,",100*H/hf_avg,"% efficiency")

# Edge detection
forward,backward,central=first_order_edges(gray)
log_resp=conv2(gray,log_kernel(9,1.4))
dog_resp=conv2(gray,gaussian_kernel(9,1.0))-conv2(gray,gaussian_kernel(9,2.0))
log_edges=zero_cross(log_resp,2.0)
dog_edges=zero_cross(dog_resp,1.0)

print("All DIP operations completed.")
