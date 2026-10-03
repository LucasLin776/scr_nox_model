from __future__ import annotations
import os, json, pickle, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
ROOT=Path('/Users/shuoqinlin/scr_nox_model')
RAW=ROOT/'raw_data.csv'
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
SEED=42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

# Stable short names grounded in the source workbook.
df=pd.read_csv(RAW, parse_dates=['日期时间'])
cols=list(df.columns)
# identify columns by index because source contains duplicate tags
c={
 'dt':cols[0], 'sec':cols[1], 'tail':cols[2], 'sp':cols[3],
 'noin_a':cols[4], 'noout_a':cols[5], 'nh3_a':cols[6], 'valve_a':cols[8],
 'noin_b':cols[9], 'noout_b':cols[10], 'nh3_b':cols[11], 'valve_b':cols[12],
 'auto':cols[13], 'air':cols[14], 'load':cols[15], 'purge':cols[16],
 'coal_a':cols[17], 'coal_b':cols[18], 'coal_c':cols[19], 'coal_d':cols[20], 'coal_e':cols[21], 'coal_f':cols[22],
 'paf_a':cols[23], 'paf_b':cols[24], 'paf_c':cols[25], 'paf_d':cols[26], 'paf_e':cols[27], 'paf_f':cols[28],
 'temp_a':cols[29], 'temp_b':cols[30]
}
use={k:v for k,v in c.items() if k not in ['dt','sec']}
base=df[[c['dt'],*use.values()]].copy()
base=base.rename(columns={c['dt']:'dt', **{v:k for k,v in use.items()}}).set_index('dt').sort_index()
# Numerical validity rules: the source uses 327.67 as an analyzer upper sentinel and negative NH3 flow as a bad value.
for k in ['tail','noout_a','noout_b']:
    base.loc[base[k]>=327.65,k]=np.nan
for k in ['nh3_a','nh3_b']:
    base.loc[base[k]<0,k]=np.nan
# Duplicate side tag already collapsed; engineer physically meaningful totals.
base['coal_total']=base[[f'coal_{x}' for x in 'abcdef']].sum(axis=1, min_count=1)
base['paf_total']=base[[f'paf_{x}' for x in 'abcdef']].sum(axis=1, min_count=1)
# Aggregate one-second signals to robust 10-second medians.
agg=base.resample('10s').median()
# Limited interpolation only for short instrument gaps, never across long outages.
num_cols=agg.columns
agg[num_cols]=agg[num_cols].interpolate(limit=3, limit_direction='both')
# Stable-operating subset used for model identification.
valid=(agg['load']>=100)&(agg['tail'].notna())&(agg['noout_a'].notna())&(agg['noout_b'].notna())&(agg['nh3_a']>=0)&(agg['nh3_b']>=0)&(agg['temp_a']>250)&(agg['temp_b']>250)&(agg['noin_a']>100)&(agg['noin_b']>100)
clean=agg.loc[valid].copy()
# Keep original timestamp and derived values.
clean.index.name='dt'
clean.to_csv(ROOT/'clean_10s.csv')

base_features=['sp','noin_a','noin_b','noout_a','noout_b','nh3_a','nh3_b','valve_a','valve_b','auto','air','load','purge','coal_total','paf_total','temp_a','temp_b']
# Add simple physical and dynamic derived terms.
clean['nh3_total']=clean['nh3_a']+clean['nh3_b']
clean['noin_mean']=(clean['noin_a']+clean['noin_b'])/2
clean['noout_mean']=(clean['noout_a']+clean['noout_b'])/2
clean['removal_eff']=(clean['noin_mean']-clean['noout_mean'])/clean['noin_mean'].clip(lower=1)
base_features += ['nh3_total','noin_mean','noout_mean','removal_eff']
# Target is tail-duct outlet NOx. Build causal lag features for tree models.
lag_list=[1,3,6,12,24]
tree_features=[]
for f in base_features:
    tree_features.append(f)
    for lag in lag_list:
        name=f'{f}_lag{lag}'; clean[name]=clean[f].shift(lag); tree_features.append(name)
for lag in lag_list:
    name=f'tail_lag{lag}'; clean[name]=clean['tail'].shift(lag); tree_features.append(name)
model=clean.dropna(subset=tree_features+['tail']).copy()
# chronological split; no random shuffling to prevent temporal leakage.
n=len(model); i1=int(n*0.70); i2=int(n*0.85)
train=model.iloc[:i1]; val=model.iloc[i1:i2]; test=model.iloc[i2:]
Xtr=train[tree_features].to_numpy(np.float32); ytr=train['tail'].to_numpy(np.float32)
Xv=val[tree_features].to_numpy(np.float32); yv=val['tail'].to_numpy(np.float32)
Xte=test[tree_features].to_numpy(np.float32); yte=test['tail'].to_numpy(np.float32)

def metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'SMAPE_%':float(np.mean(2*np.abs(e)/(np.abs(y)+np.abs(p)+1e-6))*100),'bias':float(np.mean(e))}
results={}
# Persistence benchmark uses the last observed tail output.
pers=test['tail_lag1'].to_numpy(); results['persistence_10s']=metrics(yte,pers)
models={
 'ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10.0))]),
 'extra_trees':ExtraTreesRegressor(n_estimators=180,max_features=0.65,min_samples_leaf=2,n_jobs=4,random_state=SEED),
 'random_forest':RandomForestRegressor(n_estimators=160,max_features=0.7,min_samples_leaf=2,n_jobs=4,random_state=SEED),
 'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=350,learning_rate=0.045,max_leaf_nodes=31,l2_regularization=0.4,random_state=SEED)
}
for name,m in models.items():
    m.fit(Xtr,ytr)
    pv=m.predict(Xv); pt=m.predict(Xte)
    results[name]={'val':metrics(yv,pv),'test':metrics(yte,pt)}
    joblib.dump(m,ART/f'{name}.joblib')
# Select best tree model by validation RMSE.
best_name=min(models,key=lambda nm:results[nm]['val']['RMSE']); best=models[best_name]
pt=best.predict(Xte)
# Feature permutation importance on a bounded subset for speed.
sub=np.arange(0,len(Xte),max(1,len(Xte)//1500))
try:
    pi=permutation_importance(best,Xte[sub],yte[sub],n_repeats=4,random_state=SEED,n_jobs=4,scoring='neg_root_mean_squared_error')
    imp=pd.DataFrame({'feature':tree_features,'importance':pi.importances_mean}).sort_values('importance',ascending=False)
    imp.to_csv(ART/'feature_importance.csv',index=False)
except Exception as e:
    imp=pd.DataFrame({'feature':tree_features,'importance':np.nan})

# GRU: causal sequence ends at t-1 and predicts tail at t.
seq_features=base_features+['tail']
arr=clean[seq_features].dropna().copy()
# Align to the same chronological endpoint proportions.
end_times=arr.index
# Use all clean sequence rows; split by endpoints.
n2=len(arr); s1=int(n2*0.70); s2=int(n2*0.85)
# standardize from train only
mu=arr.iloc[:s1][seq_features].mean().to_numpy(np.float32); sd=arr.iloc[:s1][seq_features].std().replace(0,1).to_numpy(np.float32)
A=((arr[seq_features].to_numpy(np.float32)-mu)/sd)
yarr=arr['tail'].to_numpy(np.float32)
# Exclude rare analyzer/switching outliers from the deep learner while retaining them in data-quality counts.
stable_tail=(yarr<=120)
seq_len=36
def make_windows(start,end):
    # endpoint t in [start,end); inputs are t-seq_len..t-1, target t
    xs=[]; ys=[]
    for t in range(max(start,seq_len),end):
        if (not stable_tail[t]) or (not stable_tail[t-seq_len:t].all()):
            continue
        xs.append(A[t-seq_len:t]); ys.append(yarr[t])
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32)
GXtr,Gytr=make_windows(0,s1); GXv,Gyv=make_windows(s1,s2); GXte,Gyte=make_windows(s2,n2)
class GRUReg(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__(); self.gru=nn.GRU(d,hidden,batch_first=True,num_layers=2,dropout=.15); self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.head(self.gru(x)[0][:,-1,:]).squeeze(-1)

device='cpu'; net=GRUReg(GXtr.shape[2]).to(device)
train_loader=DataLoader(TensorDataset(torch.tensor(GXtr),torch.tensor(Gytr)),batch_size=128,shuffle=False)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-4); loss_fn=nn.SmoothL1Loss()
best_state=None; best_val=float('inf'); patience=0
for epoch in range(1,61):
    net.train(); total=0
    for xb,yb in train_loader:
        opt.zero_grad(); pred=net(xb); loss=loss_fn(pred,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); total+=loss.item()*len(xb)
    net.eval()
    with torch.no_grad(): vp=net(torch.tensor(GXv)).numpy()
    vr=metrics(Gyv,vp)['RMSE']
    if vr<best_val:
        best_val=vr; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; patience=0
    else: patience+=1
    if patience>=10: break
net.load_state_dict(best_state); net.eval()
with torch.no_grad(): gp=net(torch.tensor(GXte)).numpy()
results['gru']={'val':metrics(Gyv,net(torch.tensor(GXv)).detach().numpy()),'test':metrics(Gyte,gp),'epochs':epoch}
torch.save({'state_dict':net.state_dict(),'mu':mu,'sd':sd,'seq_features':seq_features,'seq_len':seq_len},ART/'gru.pt')
# Save predictions for plotting / UI.
pred_df=pd.DataFrame({'dt':test.index.astype(str),'actual_tail_nox':yte,'persistence':pers,'ridge':models['ridge'].predict(Xte),'extra_trees':models['extra_trees'].predict(Xte),'random_forest':models['random_forest'].predict(Xte),'hist_gradient_boosting':models['hist_gradient_boosting'].predict(Xte)})
# GRU predictions have slightly shorter endpoint range; save separately.
pred_df.to_csv(ART/'test_predictions.csv',index=False)
pd.DataFrame({'dt':arr.index[s2+seq_len if False else s2:].astype(str)[:len(gp)],'actual_tail_nox':Gyte,'gru':gp}).to_csv(ART/'gru_test_predictions.csv',index=False)
# Save metadata for API and report.
metadata={'source_file':str(RAW.name),'raw_rows':int(len(df)),'aggregated_rows':int(len(agg)),'clean_rows':int(len(clean)),'model_rows':int(len(model)),'valid_fraction':float(len(clean)/len(agg)),'resample_seconds':10,'target':'tail (尾部烟道出口 NOx, mg/Nm3)','best_tree_model':best_name,'base_features':base_features,'tree_features':tree_features,'seq_features':seq_features,'split':{'train':int(len(train)),'validation':int(len(val)),'test':int(len(test))},'cleaning_rules':['outputs >=327.65 treated as analyzer sentinel','negative NH3 flows treated as invalid','10-second median aggregation','stable load >=100 MW and valid temperature/inlet/outlet signals']}
with open(ART/'metrics.json','w') as f: json.dump(results,f,ensure_ascii=False,indent=2)
with open(ART/'metadata.json','w') as f: json.dump(metadata,f,ensure_ascii=False,indent=2)
print(json.dumps({'metadata':metadata,'metrics':results,'top_features':imp.head(20).to_dict('records')},ensure_ascii=False,indent=2))
