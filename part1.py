import numpy as np
from common import load_data,write_results
_,X,y,value,_=load_data()
n=len(value); x=X[:,3]
mean=value.mean(); sd=value.std(ddof=1); se=sd/np.sqrt(n)
low=mean-1.96*se; high=mean+1.96*se
z=(mean-8.0)/se; reject=abs(z)>1.96
nreq=int(np.ceil((1.96*sd/0.5)**2))
xm=x.mean(); ym=value.mean()
slope=np.sum((x-xm)*(value-ym))/np.sum((x-xm)**2)
inter=ym-slope*xm; fit=inter+slope*x; resid=value-fit
tss=np.sum((value-ym)**2); ess=np.sum((fit-ym)**2); rss=np.sum(resid**2)
r2=1-rss/tss; rmse=np.sqrt(rss/n); mae=np.mean(np.abs(resid))
xs=(x-xm)/x.std(ddof=1); m=c=0.0
def cost(mm,cc): return float(np.mean((value-(mm*xs+cc))**2))
gd={0:cost(0,0)}
for ep in range(1,501):
    e=value-(m*xs+c); gm=-(2/n)*np.sum(xs*e); gc=-(2/n)*np.sum(e)
    m-=0.1*gm; c-=0.1*gc
    if ep in (1,10,100,500): gd[ep]=cost(m,c)
gds=m/x.std(ddof=1); gdi=c-m*xm/x.std(ddof=1)
order=np.argsort(fit,kind="stable")
lo_sd=np.std(resid[order[:400]],ddof=1); hi_sd=np.std(resid[order[400:]],ddof=1)
lx=np.log(x); ly=np.log(value); lxm=lx.mean(); lym=ly.mean()
ls=np.sum((lx-lxm)*(ly-lym))/np.sum((lx-lxm)**2); li=lym-ls*lxm
lf=li+ls*lx; lr=ly-lf; ltss=np.sum((ly-lym)**2); lrss=np.sum(lr**2)
lr2=1-lrss/ltss; lrmse=np.sqrt(lrss/n); lo=np.argsort(lf,kind="stable")
llsd=np.std(lr[lo[:400]],ddof=1); lhusd=np.std(lr[lo[400:]],ddof=1)
M=np.column_stack([np.ones(n),np.log(x),X[:,[0,1,2,4,5,6]]])
beta=np.linalg.solve(M.T@M,M.T@np.log(value))
pred=M@beta; my=np.log(value).mean(); mrss=np.sum((np.log(value)-pred)**2); mtss=np.sum((np.log(value)-my)**2)
mr2=1-mrss/mtss; mradj=1-(1-mr2)*(n-1)/(n-7-1); ladj=1-(1-lr2)*(n-1)/(n-1-1)
items=[("n",n),("mean_value",mean),("sd_value",sd),("se_value",se),("ci95_low",low),("ci95_high",high),("z_stat",z),("reject_h0",reject),("n_required",nreq),("reg_slope",slope),("reg_intercept",inter),("tss",tss),("ess",ess),("rss",rss),("tss_equals_ess_plus_rss",abs(tss-ess-rss)<1e-6*tss),("r2",r2),("rmse",rmse),("mae",mae),("gd_J0",gd[0]),("gd_J1",gd[1]),("gd_J10",gd[10]),("gd_J100",gd[100]),("gd_J500",gd[500]),("gd_slope",gds),("gd_intercept",gdi),("gd_slope_gap",abs(gds-slope)),("resid_sd_lower_raw",lo_sd),("resid_sd_upper_raw",hi_sd),("resid_sd_ratio_raw",hi_sd/lo_sd),("loglog_intercept",li),("loglog_slope",ls),("loglog_r2",lr2),("loglog_rmse",lrmse),("resid_sd_lower_log",llsd),("resid_sd_upper_log",lhusd),("resid_sd_ratio_log",lhusd/llsd)]
items += [(f"mr_beta_{i}",v) for i,v in enumerate(beta)]
items += [("mr_r2",mr2),("mr_adj_r2",mradj),("loglog_adj_r2",ladj)]
write_results("outputs/part1_results.txt",items)
