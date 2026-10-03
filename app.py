from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os, json
import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent; ART=ROOT/'artifacts'; STATIC=ROOT/'static'
app=FastAPI(title='SCR 脱硝数据驱动建模与控制',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
app.mount('/artifacts', StaticFiles(directory=str(ART)), name='artifacts')
metrics=json.load(open(ART/'metrics.json',encoding='utf-8')); meta=json.load(open(ART/'metadata.json',encoding='utf-8')); quality=json.load(open(ART/'data_quality.json',encoding='utf-8')); control=json.load(open(ART/'control_recommendation.json',encoding='utf-8'))
imp=pd.read_csv(ART/'feature_importance.csv').head(20)
clean=pd.read_csv(ROOT/'clean_10s.csv',parse_dates=['dt'])
rf=joblib.load(ART/'random_forest.joblib')
base=['sp','noin_a','noin_b','noout_a','noout_b','nh3_a','nh3_b','valve_a','valve_b','auto','air','load','purge','coal_total','paf_total','temp_a','temp_b']
clean['nh3_total']=clean.nh3_a+clean.nh3_b; clean['noin_mean']=(clean.noin_a+clean.noin_b)/2; clean['noout_mean']=(clean.noout_a+clean.noout_b)/2; clean['removal_eff']=(clean.noin_mean-clean.noout_mean)/clean.noin_mean.clip(lower=1)
base += ['nh3_total','noin_mean','noout_mean','removal_eff']; lags=[1,3,6,12,24]; feats=[]
for f in base:
    feats.append(f)
    for l in lags:
        clean[f'{f}_lag{l}']=clean[f].shift(l); feats.append(f'{f}_lag{l}')
for l in lags:
    clean[f'tail_lag{l}']=clean['tail'].shift(l); feats.append(f'tail_lag{l}')
model_df=clean.dropna(subset=feats+['tail']).copy()

class ControlRequest(BaseModel):
    multiplier_min: float=0.80
    multiplier_max: float=1.20
    margin_mg: float=2.0

def feature_row(row): return np.asarray([[float(row[f]) for f in feats]],dtype=np.float32)
def candidate_control(row,mn=.8,mx=1.2,margin=2.0):
    cur_a=float(row.nh3_a); cur_b=float(row.nh3_b); sp=float(row.sp); items=[]
    for ma in np.linspace(mn,mx,25):
        for mb in np.linspace(mn,mx,25):
            z=row.copy(); z['nh3_a']=cur_a*ma; z['nh3_b']=cur_b*mb; z['nh3_total']=z.nh3_a+z.nh3_b
            pred=float(rf.predict(feature_row(z))[0]); total=float(z.nh3_a+z.nh3_b)
            items.append({'mult_a':float(ma),'mult_b':float(mb),'nh3_a':float(z.nh3_a),'nh3_b':float(z.nh3_b),'total_nh3':total,'pred_tail_nox':pred,'target_setpoint':sp,'meets_setpoint':bool(pred<=sp+margin)})
    allc=pd.DataFrame(items); feasible=allc[allc.meets_setpoint]
    choice=(feasible.sort_values(['total_nh3','pred_tail_nox']).iloc[0] if len(feasible) else allc.sort_values(['pred_tail_nox','total_nh3']).iloc[0])
    return {'current':{'dt':str(row['dt']),'load_mw':float(row.load),'setpoint':sp,'nh3_a':cur_a,'nh3_b':cur_b,'predicted_tail_nox':float(rf.predict(feature_row(row))[0])},'recommendation':choice.to_dict(),'candidate_count':len(allc),'feasible_count':len(feasible),'constraint':f'predicted tail NOx <= setpoint + {margin} mg/Nm3; NH3 multipliers {mn:.2f}–{mx:.2f}'}

@app.get('/')
def home(): return FileResponse(STATIC/'index.html')
@app.get('/api/summary')
def summary():
    t=clean.tail(720).copy();
    # downsample for browser payload
    t=t.iloc[::max(1,len(t)//360)]
    series=[]
    for _,r in t.iterrows(): series.append({'dt':str(r['dt']),'tail_nox':float(r['tail']),'setpoint':float(r.sp),'load':float(r.load),'nh3_a':float(r.nh3_a),'nh3_b':float(r.nh3_b),'noin_a':float(r.noin_a),'noin_b':float(r.noin_b),'noout_a':float(r.noout_a),'noout_b':float(r.noout_b),'temp_a':float(r.temp_a),'temp_b':float(r.temp_b)})
    return {'metadata':meta,'quality':quality,'metrics':metrics,'importance':imp.to_dict('records'),'control':control,'series':series}
@app.get('/api/predict')
def predict():
    row=model_df.iloc[-1]; p=float(rf.predict(feature_row(row))[0]); return {'dt':str(row['dt']),'prediction_tail_nox':p,'actual_tail_nox':float(row['tail']),'setpoint':float(row.sp),'model':meta['best_tree_model']}
@app.post('/api/control')
def control_api(req:ControlRequest):
    row=model_df.iloc[-1]
    if req.multiplier_min<=0 or req.multiplier_max<req.multiplier_min: raise HTTPException(400,'invalid multiplier range')
    return candidate_control(row,req.multiplier_min,req.multiplier_max,req.margin_mg)
class DeepSeekRequest(BaseModel): question: str
@app.post('/api/deepseek')
def deepseek(req:DeepSeekRequest):
    key=os.getenv('DEEPSEEK_API_KEY')
    if not key: return {'ok':False,'message':'未配置 DEEPSEEK_API_KEY；请在后端环境变量中配置后再调用。','fallback':'可先查看本地模型指标、变量重要性和控制推荐。'}
    context={'metrics':metrics,'metadata':meta,'top_features':imp.head(10).to_dict('records'),'control':control,'quality':quality}
    body={'model':os.getenv('DEEPSEEK_MODEL','deepseek-chat'),'messages':[{'role':'system','content':'你是电厂SCR脱硝建模助手。只根据给定统计摘要回答，明确区分数据事实、模型推断和需要现场确认的建议，不要编造未提供的工艺参数。'}, {'role':'user','content':req.question+'\n统计摘要：'+json.dumps(context,ensure_ascii=False)}], 'temperature':0.2}
    try:
        r=requests.post(os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com/chat/completions'),headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json=body,timeout=60); r.raise_for_status(); data=r.json(); return {'ok':True,'answer':data['choices'][0]['message']['content'],'model':body['model']}
    except Exception as e: return {'ok':False,'message':f'DeepSeek API 调用失败：{e}'}
