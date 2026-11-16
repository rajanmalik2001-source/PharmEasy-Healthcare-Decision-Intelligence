import numpy as np
from common import load_data,write_results,write_csv,stratified_split,standardize_fit_apply,sigmoid,fit_logistic,knn_probs,knn_cv,confusion
rows,X,y,value,_=load_data(); tr,te=stratified_split(y)
def entropy(v):
    if len(v)==0:return 0.0
    p=np.mean(v); return -(p*np.log2(p)+(1-p)*np.log2(1-p)) if 0<p<1 else 0.0
def gini(v):
    if len(v)==0:return 0.0
    p=np.mean(v); return 1-p*p-(1-p)*(1-p)
yt=y[tr]; root_e=entropy(yt); root_g=gini(yt)
conds=[X[tr,5]==1,X[tr,6]==1,X[tr,2]>0.5,X[tr,0]<=24]
splitrows=[]; gains=[]
for name,cond in zip(("S1","S2","S3","S4"),conds):
    yes=yt[cond]; no=yt[~cond]
    we=(len(yes)*entropy(yes)+len(no)*entropy(no))/len(yt)
    wg=(len(yes)*gini(yes)+len(no)*gini(no))/len(yt)
    ig=root_e-we; gg=root_g-wg
    splitrows.append((name,len(yes),int(yes.sum()),len(no),int(no.sum()),ig,gg)); gains.append((name,ig,gg))
best_i=max(gains,key=lambda z:z[1])[0]; best_g=max(gains,key=lambda z:z[2])[0]
best_idx=("S1","S2","S3","S4").index(best_i); ctrain=conds[best_idx]
stump_yes=float(yt[ctrain].mean()); stump_no=float(yt[~ctrain].mean())
Z,mu,sd=standardize_fit_apply(X,tr); w,b,_=fit_logistic(Z[tr],yt,0.1,1000,())
prob_log=sigmoid(Z[te]@w+b); cv,bestk=knn_cv(Z[tr],yt); prob_knn=knn_probs(Z[tr],yt,Z[te],bestk)
test_cond=[X[te,5]==1,X[te,6]==1,X[te,2]>0.5,X[te,0]<=24][best_idx]
prob_stump=np.where(test_cond,stump_yes,stump_no)
preds={"logistic":(prob_log>=0.3).astype(int),"knn":(prob_knn>=((bestk+1)/2)/bestk).astype(int),"stump":(prob_stump>=0.3).astype(int)}
preds["hard_vote"]=((preds["logistic"]+preds["knn"]+preds["stump"])>=2).astype(int)
preds["soft_vote"]=(((prob_log+prob_knn+prob_stump)/3)>=0.3).astype(int)
voting=[]
for name in ("logistic","knn","stump","hard_vote","soft_vote"):
    c=confusion(y[te],preds[name]); voting.append((name,c[0],c[1],c[2],c[3],c[5],c[6],c[7]))
write_csv("outputs/part4_splits.csv",["split","n_yes","positives_yes","n_no","positives_no","info_gain","gini_gain"],splitrows)
write_csv("outputs/part4_voting.csv",["model","tp","fp","tn","fn","precision","recall","f1"],voting)
yy=np.array([1,1,1,-1,-1,-1,1,-1]); h1=np.array([1,1,-1,-1,-1,-1,-1,-1]); h2=np.array([-1,1,1,-1,1,-1,1,-1])
weights=np.ones(8)/8; adarows=[]; errs=[]; alphas=[]
for rnd,h in ((1,h1),(2,h2)):
    before=weights.copy(); err=float(np.sum(before[h!=yy])); alpha=0.5*np.log((1-err)/err)
    un=before*np.exp(-alpha*yy*h); after=un/un.sum()
    errs.append(err); alphas.append(alpha)
    for i in range(8): adarows.append((rnd,i+1,before[i],un[i],after[i]))
    weights=after
ensemble=np.sign(alphas[0]*h1+alphas[1]*h2); ensemble_acc=float(np.mean(ensemble==yy))
write_csv("outputs/part4_adaboost.csv",["round","point","weight_before","weight_unnormalised","weight_after"],adarows)
arch=[("mlp_hidden1",784*16+16),("mlp_hidden2",16*16+16),("mlp_output",16*10+10),("cnn_conv1",(5*5*1+1)*6),("cnn_conv2",(5*5*6+1)*16),("cnn_dense1",(5*5*16)*32+32),("cnn_dense2",32*10+10)]
write_csv("outputs/part4_architecture.csv",["layer","parameters"],arch)
mlpt=sum(v for k,v in arch[:3]); cnnt=sum(v for k,v in arch[3:])
fc=128*128*3*1024+1024; conv=(3*3*3+1)*64; rnn=128*64+128*128+128+16*128+16
ptr=sigmoid(Z[tr]@w+b); costs=[]
for k in range(1,20):
    tau=k/20; c=confusion(yt,(ptr>=tau).astype(int)); costs.append((50*c[1]+400*c[3],tau))
cost_train,tau=min(costs,key=lambda q:(q[0],q[1]))
ct=confusion(y[te],(prob_log>=tau).astype(int)); c05=confusion(y[te],(prob_log>=0.5).astype(int))
items=[("root_entropy",root_e),("root_gini",root_g),("best_split_by_info_gain",best_i),("best_split_by_gini_gain",best_g),("stump_rate_yes",stump_yes),("stump_rate_no",stump_no),("round1_error",errs[0]),("round1_alpha",alphas[0]),("round2_error",errs[1]),("round2_alpha",alphas[1]),("ensemble_accuracy",ensemble_acc),("oob_fraction",(1-1/560)**560),("mlp_total",mlpt),("cnn_total",cnnt),("fc_layer_params",fc),("conv_layer_params",conv),("rnn_total",rnn),("cost_tau",tau),("cost_test_at_tau",50*ct[1]+400*ct[3]),("cost_test_at_0.5",50*c05[1]+400*c05[3])]
write_results("outputs/part4_results.txt",items)
