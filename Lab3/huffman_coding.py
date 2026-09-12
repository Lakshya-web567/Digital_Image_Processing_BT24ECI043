import heapq

class Node:
    def __init__(self,p,symbol=None,left=None,right=None):
        self.p=p; self.symbol=symbol; self.left=left; self.right=right

def huffman_codes(symbol_probs):
    heap=[]; counter=0
    for s,p in symbol_probs:
        heapq.heappush(heap,(p,counter,Node(p,symbol=s))); counter+=1
    while len(heap)>1:
        p1,_,n1=heapq.heappop(heap); p2,_,n2=heapq.heappop(heap)
        parent=Node(p1+p2,left=n1,right=n2)
        heapq.heappush(heap,(parent.p,counter,parent)); counter+=1
    root=heap[0][2]; codes={}
    def walk(n,c):
        if n.symbol is not None: codes[n.symbol]=c or "0"; return
        walk(n.left,c+"0"); walk(n.right,c+"1")
    walk(root,"")
    return codes
