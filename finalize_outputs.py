import json, os, math
from pathlib import Path
import numpy as np, pandas as pd
import joblib
import matplotlib.pyplot as plt
ROOT=Path('/Users/shuoqinlin/scr_nox_model'); ART=ROOT/'artifacts'
metrics=json.load(open(ART/'metrics.json',encoding='utf-8'))
rf=pd.read_csv(ART/'test_predictions.csv'); gr=pd.read_csv(ART/'gru_test_predictions.csv')
m=rf.merge(gr,on=['dt','actual_tail_nox'])
y=m.actual_tail_nox.to_numpy(); rf_p=m.random_forest.to_numpy(); gru_p=m.gru.to_numpy(); fusion=.95*rf_p+.05*gru_p

def met(y,p):
 e=p-y; return {'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'R2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),'SMAPE_%':float(np.mean(2*abs(e)/(abs(y)+abs(p)+1e-6))*100),'bias':float(e.mean())}
metrics['fusion_95rf_05gru']={'test':met(y,fusion),'weights':{'random_forest':.95,'gru':.05},'note':'固定权重用于保留时序模型通道；最终控制优化仍以随机森林作为主预测器'}
m['fusion']=fusion; m.to_csv(ART/'ensemble_test_predictions.csv',index=False)
with open(ART/'metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
# Data quality summary
d=pd.read_csv(ROOT/'raw_data.csv',parse_dates=['日期时间'])
quality={'raw_rows':len(d),'time_start':str(d.iloc[:,0].min()),'time_end':str(d.iloc[:,0].max()),'duration_hours':float((d.iloc[-1,0]-d.iloc[0,0]).total_seconds()/3600),'duplicate_header_columns':['AI_NOXA_AMMFLOW','DI_NOXA_IN_MAIN'],'sentinel_counts':{'tail_327_67':int((d.iloc[:,2]>=327.65).sum()),'noout_a_327_67':int((d.iloc[:,5]>=327.65).sum()),'noout_b_327_67':int((d.iloc[:,10]>=327.65).sum()),'nh3_a_negative':int((d.iloc[:,6]<0).sum()),'nh3_b_negative':int((d.iloc[:,11]<0).sum())},'load_bins':{}}
for lo,hi in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600)]:
 x=d[(d.iloc[:,15]>=lo)&(d.iloc[:,15]<hi)]; quality['load_bins'][f'{lo}-{hi}']={'rows':len(x),'tail_mean':float(x.iloc[:,2].mean()),'tail_valid_fraction':float((x.iloc[:,2]<327.65).mean()) if len(x) else None}
json.dump(quality,open(ART/'data_quality.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
# Charts
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(12,4)); q=rf.tail(900); ax.plot(q.index,q.actual_tail_nox,label='实际值',lw=1.4); ax.plot(q.index,q.random_forest,label='随机森林',lw=1.1); ax.plot(q.index,q.hist_gradient_boosting,label='梯度提升',lw=.9,alpha=.7); ax.set_title('测试集尾部烟道出口 NOx 预测'); ax.set_xlabel('测试集样本（10 s 聚合）'); ax.set_ylabel('NOx (mg/Nm³)'); ax.legend(); fig.tight_layout(); fig.savefig(ART/'prediction_test.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5)); imp=pd.read_csv(ART/'feature_importance.csv').head(15).sort_values('importance'); ax.barh(imp.feature,imp.importance,color='#2b6cb0'); ax.set_title('随机森林置换重要性 Top 15'); ax.set_xlabel('RMSE 增量'); fig.tight_layout(); fig.savefig(ART/'feature_importance.png',dpi=160); plt.close(fig)
# Control scenario using latest row and RF.
clean=pd.read_csv(ROOT/'clean_10s.csv',parse_dates=['dt'])
# reconstruct current feature vector consistent with training
base=['sp','noin_a','noin_b','noout_a','noout_b','nh3_a','nh3_b','valve_a','valve_b','auto','air','load','purge','coal_total','paf_total','temp_a','temp_b']
clean['nh3_total']=clean.nh3_a+clean.nh3_b; clean['noin_mean']=(clean.noin_a+clean.noin_b)/2; clean['noout_mean']=(clean.noout_a+clean.noout_b)/2; clean['removal_eff']=(clean.noin_mean-clean.noout_mean)/clean.noin_mean.clip(lower=1)
base+=['nh3_total','noin_mean','noout_mean','removal_eff']; lags=[1,3,6,12,24]; feats=[]
for f in base:
 feats.append(f)
 for l in lags: clean[f'{f}_lag{l}']=clean[f].shift(l); feats.append(f'{f}_lag{l}')
for l in lags: clean[f'tail_lag{l}']=clean['tail'].shift(l); feats.append(f'tail_lag{l}')
row=clean.dropna(subset=feats+['tail']).iloc[-1].copy(); rf_model=joblib.load(ART/'random_forest.joblib')
cur_a=float(row.nh3_a); cur_b=float(row.nh3_b); sp=float(row.sp)
sc=[]
for ma in np.linspace(.80,1.20,25):
 for mb in np.linspace(.80,1.20,25):
  z=row.copy(); z['nh3_a']=cur_a*ma; z['nh3_b']=cur_b*mb; z['nh3_total']=z.nh3_a+z.nh3_b
  vals=np.array([[z[f] for f in feats]],dtype=np.float32); pred=float(rf_model.predict(vals)[0]); total=float(z.nh3_a+z.nh3_b)
  sc.append({'mult_a':float(ma),'mult_b':float(mb),'nh3_a':float(z.nh3_a),'nh3_b':float(z.nh3_b),'total_nh3':total,'pred_tail_nox':pred,'target_setpoint':sp,'meets_setpoint':bool(pred<=sp+2.0)})
sc=pd.DataFrame(sc); feasible=sc[sc.meets_setpoint]
choice=(feasible.sort_values(['total_nh3','pred_tail_nox']).iloc[0] if len(feasible) else sc.sort_values(['pred_tail_nox','total_nh3']).iloc[0])
control={'current':{'dt':str(row['dt']),'load_mw':float(row.load),'setpoint':sp,'nh3_a':cur_a,'nh3_b':cur_b,'predicted_tail_nox':float(rf_model.predict(np.array([[row[f] for f in feats]],np.float32))[0])},'recommendation':choice.to_dict(),'candidate_count':len(sc),'feasible_count':len(feasible),'constraint':'predicted tail NOx <= setpoint + 2 mg/Nm3; NH3 multipliers 0.80–1.20'}
json.dump(control,open(ART/'control_recommendation.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
# Report markdown
meta=json.load(open(ART/'metadata.json',encoding='utf-8'))
imp=pd.read_csv(ART/'feature_importance.csv').head(10)
report=f'''# 基于 DeepSeek 辅助的 600 MW 机组 SCR 脱硝过程数据驱动建模与控制\n\n## 摘要\n\n本项目使用靖海电厂 1 号 600 MW 机组脱硝实测数据，完成数据清洗、10 s 时间聚合、动态滞后特征构造、传统机器学习、深度学习和约束喷氨优化。研究对象为尾部烟道出口 NOx 浓度，输入包含 SCR 入口/出口 NOx、A/B 侧氨气流量与阀门指令、机组负荷、风量、燃料量、一次风量、入口温度和运行状态。\n\n## 1. 任务理解与工艺对象\n\nSCR 是大时延、强耦合、非线性过程。主要控制通道为 A/B 侧喷氨流量或阀门指令，主要扰动通道为机组负荷、烟气流量、入口 NOx、入口温度和燃料/一次风量。控制目标是在满足尾部 NOx 排放约束的同时，减少不必要的氨耗。\n\n## 2. 数据与预处理\n\n- 原始记录：{meta['raw_rows']:,} 行，1 s 采样，时间范围为 {quality['time_start']} 至 {quality['time_end']}。\n- 预处理：将 327.67 视为分析仪上限哨兵值；负氨流量视为无效；重复标签合并；对数值信号按 10 s 取中位数；仅使用负荷不低于 100 MW、入口/出口信号和温度有效的稳定工况。\n- 清洗后：{meta['clean_rows']:,} 个 10 s 样本，占聚合数据 {meta['valid_fraction']*100:.2f}%。\n- 数据划分：按时间顺序 70% 训练、15% 验证、15% 测试，不随机打乱，防止时间泄漏。\n\n## 3. 建模方法\n\n### 3.1 传统机器学习\n\n建立岭回归、ExtraTrees、随机森林和 HistGradientBoosting 基线。随机森林使用 160 棵树、滞后 10/30/60/120/240 s 特征，目标为尾部烟道出口 NOx。\n\n### 3.2 深度学习\n\n建立双层 GRU，使用最近 360 s 的 10 s 序列窗口预测下一时刻 NOx。GRU 训练时排除了少量尾部 NOx 大于 120 mg/Nm3 的启动/切换异常窗口，以提高稳定工况预测的鲁棒性。\n\n### 3.3 融合\n\n最终保留随机森林主通道和 GRU 时序通道，使用 95% 随机森林 + 5% GRU 的保守加权融合，并将随机森林作为控制优化的主预测器。\n\n## 4. 测试集结果\n\n| 模型 | MAE | RMSE | R² | SMAPE |\n|---|---:|---:|---:|---:|\n| 持续性基线 | {metrics['persistence_10s']['MAE']:.3f} | {metrics['persistence_10s']['RMSE']:.3f} | {metrics['persistence_10s']['R2']:.4f} | {metrics['persistence_10s']['SMAPE_%']:.2f}% |\n| 岭回归 | {metrics['ridge']['test']['MAE']:.3f} | {metrics['ridge']['test']['RMSE']:.3f} | {metrics['ridge']['test']['R2']:.4f} | {metrics['ridge']['test']['SMAPE_%']:.2f}% |\n| ExtraTrees | {metrics['extra_trees']['test']['MAE']:.3f} | {metrics['extra_trees']['test']['RMSE']:.3f} | {metrics['extra_trees']['test']['R2']:.4f} | {metrics['extra_trees']['test']['SMAPE_%']:.2f}% |\n| 随机森林 | {metrics['random_forest']['test']['MAE']:.3f} | {metrics['random_forest']['test']['RMSE']:.3f} | {metrics['random_forest']['test']['R2']:.4f} | {metrics['random_forest']['test']['SMAPE_%']:.2f}% |\n| 双层 GRU | {metrics['gru']['test']['MAE']:.3f} | {metrics['gru']['test']['RMSE']:.3f} | {metrics['gru']['test']['R2']:.4f} | {metrics['gru']['test']['SMAPE_%']:.2f}% |\n| 95/5 融合 | {metrics['fusion_95rf_05gru']['test']['MAE']:.3f} | {metrics['fusion_95rf_05gru']['test']['RMSE']:.3f} | {metrics['fusion_95rf_05gru']['test']['R2']:.4f} | {metrics['fusion_95rf_05gru']['test']['SMAPE_%']:.2f}% |\n\n随机森林在当前稳定工况测试集上表现最好，测试集 RMSE 为 {metrics['random_forest']['test']['RMSE']:.3f} mg/Nm3，R² 为 {metrics['random_forest']['test']['R2']:.4f}。持续性基线已较强，说明 10 s 聚合后的尾部 NOx 具有明显惯性；因此模型结果应理解为短时预测与控制辅助，而不是替代分析仪。\n\n![预测结果](artifacts/prediction_test.png)\n\n## 5. 变量重要性\n\n置换重要性前十项为：\n\n{chr(10).join(f'- `{r.feature}`：{r.importance:.4f}' for r in imp.itertuples())}\n\n其中尾部 NOx 的 10 s/30 s 历史值、A/B 侧出口 NOx 的历史值和脱硝效率历史值最重要，符合 SCR 过程的大时延和惯性特征。\n\n![特征重要性](artifacts/feature_importance.png)\n\n## 6. 约束喷氨优化\n\n以随机森林为数字孪生预测器，在当前 A/B 侧氨流量的 0.80–1.20 倍范围内进行网格搜索，约束预测尾部 NOx 不高于设定值加 2 mg/Nm3，并以总氨流量最小为目标。当前时刻的离线推荐结果保存在 `artifacts/control_recommendation.json`，网页可交互重算。该结果仅用于离线分析，正式投运前应加入氨逃逸、阀门速率、最小喷氨量和联锁条件。\n\n## 7. DeepSeek API 的使用方式\n\n数值清洗、模型训练、指标和控制优化全部由本地程序完成。DeepSeek API 只接收脱敏后的统计摘要、特征重要性和模型指标，用于生成数据质量解释、异常工况说明和运行建议。配置 `DEEPSEEK_API_KEY` 后可在网页中调用 `/api/deepseek`；密钥只保存在后端环境变量中。\n\n## 8. 工艺与控制框图\n\n```mermaid\nflowchart LR\n  F[负荷/给煤/一次风量] --> B[锅炉烟气生成]\n  B --> I[SCR入口 NOx、烟气温度、烟气流量]\n  NH3[A/B侧喷氨流量和阀门指令] --> R[SCR反应器与催化剂]\n  I --> R\n  R --> O[A/B侧SCR出口 NOx]\n  O --> T[尾部烟道出口 NOx]\n  T --> C[预测模型与约束优化]\n  C --> NH3\n```\n\n## 9. 结论与限制\n\n本项目完成了课程设计要求中的数据分析、数据对齐、预处理、神经网络/机器学习建模、精度评价、工艺变量分析和控制辅助设计。当前数据为单台机组约 30 h 的 1 s 记录，且存在分析仪上限哨兵值和启停/切换段，因此结论主要适用于负荷不低于 100 MW 的稳定工况。后续若要上线控制，应补充氨逃逸实测值、催化剂活性、烟气流量标定和多时间段数据，并在 Simulink 或实时仿真平台做闭环和抗扰验证。\n'''
open(ROOT/'report.md','w',encoding='utf-8').write(report)
print(json.dumps({'metrics':metrics,'control':control,'quality':quality},ensure_ascii=False,indent=2))
