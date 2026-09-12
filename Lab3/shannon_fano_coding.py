import numpy as np

def entropy(probabilities):
    p=np.asarray(probabilities,dtype=float)
    return float(-np.sum(p*np.log2(p)))

def shannon_fano_codes(symbol_probs):
    pairs=sorted(symbol_probs,key=lambda x:x[1],reverse=True)
    codes={s:"" for s,_ in pairs}
    def split(items):
        if len(items)<=1:return
        total=sum(p for _,p in items); cumulative=0; best_i=1; best_diff=float("inf")
        for i in range(1,len(items)):
            cumulative+=items[i-1][1]
            diff=abs(total-2*cumulative)
            if diff<best_diff: best_diff=diff; best_i=i
        left,right=items[:best_i],items[best_i:]
        for s,_ in left: codes[s]+="0"
        for s,_ in right: codes[s]+="1"
        split(left);split(right)
    split(pairs)
    return codes
