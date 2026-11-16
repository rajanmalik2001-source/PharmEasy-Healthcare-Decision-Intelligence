import csv
import numpy as np
def load_data():
    rows=list(csv.DictReader(open("data/records.csv",newline="")))
    X=np.array([[float(r[k]) for k in ["months_registered","missed_visit_count_12m","copay_burden_ratio","avg_bill_inr_thousands","adherence_score","chronic_condition_flag","insured_flag"]] for r in rows],float)
    y=np.array([int(r["no_show"]) for r in rows],int)
    value=np.array([float(r["care_plan_cost_inr_thousands"]) for r in rows],float)
    ids=[r["record_id"] for r in rows]
    return rows,X,y,value,ids
def write_results(path,items):
    with open(path,"w") as f:
        for k,v in items:
            if isinstance(v,(bool,np.bool_)): s="yes" if v else "no"
            elif isinstance(v,(int,np.integer)): s=str(int(v))
            elif isinstance(v,str): s=v
            else: s=f"{float(v):.4f}"
            f.write(f"{k} = {s}\n")
def write_csv(path,header,rows):
    with open(path,"w",newline="") as f:
        w=csv.writer(f,lineterminator="\n"); w.writerow(header)
        for row in rows: w.writerow([f"{x:.4f}" if isinstance(x,(float,np.floating)) else x for x in row])
def stratified_split(y):
    tr=[]; te=[]
    for cls in (0,1):
        ix=np.where(y==cls)[0]; cut=(7*len(ix))//10
        tr.extend(ix[:cut]); te.extend(ix[cut:])
    return np.array(sorted(tr)),np.array(sorted(te))
def standardize_fit_apply(X,tr):
    mu=X[tr].mean(axis=0); sd=X[tr].std(axis=0,ddof=1)
    return (X-mu)/sd,mu,sd
def sigmoid(z):
    z=np.clip(z,-500,500); return 1/(1+np.exp(-z))
def bce(y,p):
    p=np.clip(p,1e-15,1-1e-15)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def fit_logistic(X,y,eta=0.1,steps=1000,keep=()):
    w=np.zeros(X.shape[1]); b=0.0; losses={}
    if 0 in keep: losses[0]=bce(y,sigmoid(X@w+b))
    for epoch in range(1,steps+1):
        p=sigmoid(X@w+b); err=p-y
        w-=eta*(X.T@err)/len(y); b-=eta*np.mean(err)
        if epoch in keep: losses[epoch]=bce(y,sigmoid(X@w+b))
    return w,b,losses
def confusion(y,pred):
    tp=int(np.sum((y==1)&(pred==1))); fp=int(np.sum((y==0)&(pred==1)))
    tn=int(np.sum((y==0)&(pred==0))); fn=int(np.sum((y==1)&(pred==0)))
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0; acc=(tp+tn)/len(y)
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    return tp,fp,tn,fn,acc,prec,rec,f1,spec
def knn_probs(refX,refy,qX,k):
    out=[]
    for q in qX:
        d=np.sqrt(np.sum((refX-q)**2,axis=1)); ix=np.argsort(d,kind="stable")[:k]
        out.append(float(np.mean(refy[ix])))
    return np.array(out)
def knn_cv(X,y):
    rows=[]; folds=np.arange(len(y))%5
    for k in range(1,22,2):
        scores=[]
        for fold in range(5):
            va=np.where(folds==fold)[0]; rr=np.where(folds!=fold)[0]
            probs=knn_probs(X[rr],y[rr],X[va],k)
            pred=(probs>=((k+1)/2)/k).astype(int)
            c=confusion(y[va],pred); scores.append(c[7])
        rows.append((k,float(np.mean(scores))))
    best=sorted(rows,key=lambda z:(-z[1],z[0]))[0][0]
    return rows,best
