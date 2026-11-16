import numpy as np
from common import load_data,write_results,write_csv,stratified_split,standardize_fit_apply,sigmoid,bce,fit_logistic,confusion,knn_probs,knn_cv
_,X,y,_,_=load_data()
tr,te=stratified_split(y); Z,mu,sd=standardize_fit_apply(X,tr)
Xt,yt=Z[tr],y[tr]; Xv,yv=Z[te],y[te]
w,b,loss=fit_logistic(Xt,yt,0.1,1000,(0,1,10,100,1000))
ptr=sigmoid(Xt@w+b); pv=sigmoid(Xv@w+b)
items=[("train_rows",len(tr)),("train_positives",int(yt.sum())),("test_rows",len(te)),("test_positives",int(yv.sum()))]
for ep in (0,1,10,100,1000): items.append((f"loss_epoch_{ep}",loss[ep]))
items += [(f"weight_X{i+1}",v) for i,v in enumerate(w)]+[("bias",b),("test_bce",bce(yv,pv))]
for eta in (0.01,0.1,1.0):
    _,_,ls=fit_logistic(Xt,yt,eta,200,(200,)); items.append((f"loss_epoch_200_eta_{eta}",ls[200]))
for i,v in enumerate(w): items.append((f"odds_ratio_X{i+1}",np.exp(v)))
for tau in (0.5,0.3):
    c=confusion(yv,(pv>=tau).astype(int)); names=("tp","fp","tn","fn","accuracy","precision","recall","f1","specificity")
    items += [(f"tau_{tau}_{nm}",v) for nm,v in zip(names,c)]
items.append(("blind_accuracy",float(np.mean(yv==0))))
threshold_rows=[]
for tau in np.arange(0.1,1.0,0.1):
    c=confusion(yv,(pv>=tau).astype(int)); tp,fp,tn,fn,*_=c
    threshold_rows.append((round(float(tau),1),tp,fp,tn,fn,tp/(tp+fn) if tp+fn else 0,fp/(fp+tn) if fp+tn else 0))
write_csv("outputs/part2_thresholds.csv",["tau","tp","fp","tn","fn","tpr","fpr"],threshold_rows)
cv,best=knn_cv(Xt,yt); write_csv("outputs/part2_knn_cv.csv",["K","mean_cv_f1"],cv)
knp=knn_probs(Xt,yt,Xv,best); knpred=(knp>=((best+1)/2)/best).astype(int); kc=confusion(yv,knpred)
items.append(("knn_best_k",best))
for nm,v in zip(("tp","fp","tn","fn","accuracy","precision","recall","f1"),kc[:8]): items.append((f"knn_{nm}",v))
costs=[]
for k in range(1,20):
    tau=k/20; c=confusion(yt,(ptr>=tau).astype(int)); costs.append((50*c[1]+400*c[3],tau))
cost_train,tau=min(costs,key=lambda z:(z[0],z[1]))
ct=confusion(yv,(pv>=tau).astype(int)); c05=confusion(yv,(pv>=0.5).astype(int))
items += [("cost_tau",tau),("cost_train_at_tau",cost_train),("cost_test_at_tau",50*ct[1]+400*ct[3]),("cost_test_at_0.5",50*c05[1]+400*c05[3]),("cost_test_flag_nobody",400*int(yv.sum()))]
write_results("outputs/part2_results.txt",items)
