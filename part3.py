import numpy as np
from common import load_data,write_results,write_csv
rows,X,y,value,_=load_data(); n=len(y); A=X[:,:5]
mu=A.mean(axis=0); sd=A.std(axis=0,ddof=1); Z=(A-mu)/sd
wcss={}
for k in range(2,7):
    step=n//k; cent=Z[np.arange(k)*step].copy(); labels=np.full(n,-1)
    while True:
        d=np.sum((Z[:,None,:]-cent[None,:,:])**2,axis=2); new=np.argmin(d,axis=1)
        if np.array_equal(new,labels): labels=new; break
        labels=new
        for j in range(k):
            if np.any(labels==j): cent[j]=Z[labels==j].mean(axis=0)
    wcss[k]=float(np.sum((Z-cent[labels])**2))
elbow=next(k for k in (3,4,5) if wcss[k]-wcss[k+1] < 0.5*(wcss[k-1]-wcss[k]))
k=3; cent=Z[np.arange(k)*(n//k)].copy(); labels=np.full(n,-1)
while True:
    d=np.sum((Z[:,None,:]-cent[None,:,:])**2,axis=2); new=np.argmin(d,axis=1)
    if np.array_equal(new,labels): labels=new; break
    labels=new
    for j in range(k):
        if np.any(labels==j): cent[j]=Z[labels==j].mean(axis=0)
clusters=sorted(range(3),key=lambda j:A[labels==j,2].mean())
segrows=[]; segids={}
for letter,j in zip("ABC",clusters):
    ix=np.where(labels==j)[0]; segids[letter]=ix
    segrows.append((letter,len(ix),*A[ix].mean(axis=0),float(y[ix].mean()),float(value[ix].mean())))
write_csv("outputs/part3_segments.csv",["segment","size","mean_X1","mean_X2","mean_X3","mean_X4","mean_X5","outcome_rate","mean_value"],segrows)
D=np.array([[0,7.9,7.4,2.9,2.9,6.1],[7.9,0,1.4,5.9,5.3,2.0],[7.4,1.4,0,5.8,4.6,1.3],[2.9,5.9,5.8,0,2.8,4.5],[2.9,5.3,4.6,2.8,0,3.4],[6.1,2.0,1.3,4.5,3.4,0]],float)
hrows=[]; cutcounts={}
for method in ("single","complete","average"):
    cl=[{i} for i in range(6)]; heights=[]
    while len(cl)>1:
        candidates=[]
        for i in range(len(cl)):
            for j in range(i+1,len(cl)):
                vals=[D[a,b] for a in cl[i] for b in cl[j]]
                dist=min(vals) if method=="single" else (max(vals) if method=="complete" else float(np.mean(vals)))
                candidates.append((dist,i,j))
        dist,i,j=min(candidates,key=lambda z:(z[0],min(cl[z[1]]),min(cl[z[2]])))
        ca,cb=cl[i],cl[j]
        if min(ca)>min(cb): ca,cb=cb,ca
        fmt=lambda s:"+".join(f"E{x+1}" for x in sorted(s))
        hrows.append((method,len(heights)+1,fmt(ca),fmt(cb),dist)); heights.append(dist)
        merged=cl[i]|cl[j]
        for ix in sorted((i,j),reverse=True): cl.pop(ix)
        cl.append(merged)
    gaps=np.diff(heights); gapidx=int(np.argmax(gaps)); cutcounts[method]=6-(gapidx+1)
write_csv("outputs/part3_hierarchical.csv",["linkage","merge_no","cluster_a","cluster_b","height"],hrows)
S=(Z.T@Z)/(n-1); vals,vecs=np.linalg.eigh(S); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
for j in range(vecs.shape[1]):
    ix=np.argmax(np.abs(vecs[:,j]))
    if vecs[ix,j]<0: vecs[:,j]*=-1
proj=Z@vecs[:,:2]; ratios=vals/vals.sum(); cum=np.cumsum(ratios)
items=[(f"wcss_k{k}",wcss[k]) for k in range(2,7)]+[("elbow_k",elbow)]
for letter,row in zip("ABC",segrows): items.append((f"size_segment_{letter}",row[1]))
items += [("single_clusters_at_cut",cutcounts["single"]),("complete_clusters_at_cut",cutcounts["complete"]),("average_clusters_at_cut",cutcounts["average"]),("eigenvalue_sum",vals.sum())]
items += [(f"eigenvalue_{i+1}",v) for i,v in enumerate(vals)]
items += [(f"explained_ratio_{i+1}",v) for i,v in enumerate(ratios)]
items += [(f"cumulative_ratio_{i+1}",v) for i,v in enumerate(cum)]
items.append(("components_for_80pct",int(np.argmax(cum>=0.8)+1)))
items += [(f"pc1_loading_X{i+1}",v) for i,v in enumerate(vecs[:,0])]
items += [(f"pc2_loading_X{i+1}",v) for i,v in enumerate(vecs[:,1])]
for i in range(3): items += [(f"row{i+1}_pc1",proj[i,0]),(f"row{i+1}_pc2",proj[i,1])]
for letter in "ABC": items.append((f"mean_pc1_segment_{letter}",float(proj[segids[letter],0].mean())))
write_results("outputs/part3_results.txt",items)
