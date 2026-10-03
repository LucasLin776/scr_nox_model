from pathlib import Path
import json, pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path('/Users/shuoqinlin/scr_nox_model'); ART=ROOT/'artifacts'
metrics=json.load(open(ART/'metrics.json',encoding='utf-8')); meta=json.load(open(ART/'metadata.json',encoding='utf-8')); quality=json.load(open(ART/'data_quality.json',encoding='utf-8')); control=json.load(open(ART/'control_recommendation.json',encoding='utf-8'))
imp=pd.read_csv(ART/'feature_importance.csv').head(10)

def set_cell_shading(cell, fill):
 tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_font(run,size=10.5,bold=False,color=None):
 run.font.name='SimSun'; run._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); run.font.size=Pt(size); run.bold=bold
 if color: run.font.color.rgb=RGBColor(*color)
def add_text(p,text,size=10.5,bold=False,color=None):
 r=p.add_run(text); set_font(r,size,bold,color); return r
def heading(doc,text,level=1):
 p=doc.add_paragraph(); p.style=f'Heading {level}'; add_text(p,text,14 if level==1 else 12,bold=True,color=(31,78,121) if level==1 else (45,45,45)); return p
def para(doc,text,align=None):
 p=doc.add_paragraph(); p.paragraph_format.line_spacing=1.35; p.paragraph_format.space_after=Pt(6); add_text(p,text,10.5); 
 if align: p.alignment=align
 return p
def table(doc, headers, rows, widths=None):
 t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
 for i,h in enumerate(headers):
  cell=t.rows[0].cells[i]; set_cell_shading(cell,'D9EAF7'); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; add_text(cell.paragraphs[0],str(h),9.5,bold=True)
 for row in rows:
  cells=t.add_row().cells
  for i,v in enumerate(row): add_text(cells[i].paragraphs[0],str(v),9.2)
 return t

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.75); sec.bottom_margin=Inches(.7); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles
styles['Normal'].font.name='SimSun'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Normal'].font.size=Pt(10.5)
# Cover
for _ in range(4): doc.add_paragraph()
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; add_text(p,'自动控制系统课程设计',22,bold=True,color=(31,78,121))
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; add_text(p,'基于 DeepSeek 辅助的电厂脱硝过程数据驱动建模与控制',18,bold=True)
for _ in range(5): doc.add_paragraph()
for label,value in [('研究对象','靖海电厂 1 号 600 MW 机组 SCR 脱硝过程'),('数据范围','2024-09-26 12:17:16 至 2024-09-27 18:17:16'),('主要方法','数据清洗 + 传统机器学习 + GRU + 约束喷氨优化'),('完成日期','2026-10-03')]:
 p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; add_text(p,f'{label}：{value}',11)
doc.add_page_break()
# abstract
heading(doc,'摘要',1)
para(doc,f'本设计使用靖海电厂 1 号 600 MW 机组脱硝实测数据，完成 SCR 脱硝过程的数据质量分析、时间对齐、动态特征构造、传统机器学习建模、深度学习建模和约束喷氨优化。原始数据包含 {meta["raw_rows"]:,} 条 1 秒记录；剔除分析仪上限哨兵值、负氨流量和不稳定低负荷片段后，形成 {meta["clean_rows"]:,} 个 10 秒稳定工况样本。随机森林在时间顺序测试集上的 RMSE 为 {metrics["random_forest"]["test"]["RMSE"]:.3f} mg/Nm³，R² 为 {metrics["random_forest"]["test"]["R2"]:.4f}；双层 GRU 用于提取时序模式，最终以 95% 随机森林和 5% GRU 组成保守融合通道。基于随机森林数字孪生模型，对 A/B 侧喷氨流量进行了带排放约束的离线优化。')
heading(doc,'关键词',2); para(doc,'SCR 脱硝；NOx 预测；随机森林；GRU；数据驱动控制；DeepSeek API')
heading(doc,'目录',1)
for x in ['1 任务背景与工艺对象','2 数据来源与预处理','3 特征工程与建模方法','4 模型精度评价','5 变量重要性与工艺分析','6 约束喷氨控制设计','7 DeepSeek API 与网页系统','8 结论与后续工作']: para(doc,x)
doc.add_page_break()
# sections
heading(doc,'1 任务背景与工艺对象',1)
para(doc,'SCR 脱硝系统属于大时延、强耦合、非线性控制对象。入口 NOx、烟气温度、烟气量和机组负荷构成主要扰动通道；A/B 侧氨气流量和阀门指令构成主要控制通道；尾部烟道出口 NOx 是最终排放指标。喷氨不足会导致 NOx 超标，喷氨过量会增加氨耗并带来氨逃逸、硫酸氢铵沉积和空预器堵塞风险。')
para(doc,'本设计的建模目标是对稳定工况下尾部烟道出口 NOx 进行短时预测，并把预测模型用于离线喷氨量搜索。预测模型使用可在线获得的负荷、入口/出口 NOx、喷氨量、温度、风量、给煤量和历史 NOx 信息。')
heading(doc,'2 数据来源与预处理',1)
para(doc,f'数据文件包含一个工作表，原始记录 {quality["raw_rows"]:,} 行、32 列，采样间隔为 1 s，覆盖 {quality["duration_hours"]:.1f} h。源表存在两个重复标签：A 侧氨流量和 A 侧反吹状态各重复一次；重复列经数值一致性检查后合并。')
table(doc,['质量项目','处理结果'],[['尾部/A出口/B出口 NOx = 327.67','视为分析仪上限哨兵值并设为缺失'],['A/B侧氨流量 < 0','视为无效测量并设为缺失'],['时间对齐','按时间索引排序，10 s 中位数聚合'],['短缺口','仅对不超过 3 个聚合点的数值缺口插值'],['稳定工况','负荷 ≥ 100 MW，温度、入口/出口 NOx 有效'],['数据划分','按时间 70%/15%/15%，不随机打乱']])
para(doc,'清洗后保留稳定工况样本占聚合数据的 {:.2f}%。低负荷和分析仪饱和段仍保留在数据质量统计中，但不用于稳定工况预测器训练。'.format(meta['valid_fraction']*100))
heading(doc,'3 特征工程与建模方法',1)
para(doc,'以尾部烟道出口 NOx 为预测目标。基础特征包括 A/B 侧入口与出口 NOx、A/B 侧氨流量和阀门指令、负荷、总风量、给煤总量、一次风总量、入口温度、设定值和运行状态。为描述 SCR 时滞，构造 10、30、60、120、240 s 滞后特征，同时构造氨总流量、入口 NOx 均值、出口 NOx 均值和脱硝效率。')
para(doc,'传统模型包含岭回归、ExtraTrees、随机森林和 HistGradientBoosting。深度模型为双层 GRU，输入最近 360 s 的 10 s 序列并预测下一个 10 s 的尾部 NOx。GRU 在训练时排除了少量尾部 NOx 大于 120 mg/Nm³ 的启动/切换窗口，避免极端异常点主导稳定工况学习。')
heading(doc,'4 模型精度评价',1)
rows=[]
for name,label in [('persistence_10s','持续性基线'),('ridge','岭回归'),('extra_trees','ExtraTrees'),('random_forest','随机森林'),('gru','双层 GRU'),('fusion_95rf_05gru','95/5 融合')]:
 m=metrics[name].get('test',metrics[name]); rows.append([label,f'{m["MAE"]:.3f}',f'{m["RMSE"]:.3f}',f'{m["R2"]:.4f}',f'{m["SMAPE_%"]:.2f}%'])
table(doc,['模型','MAE','RMSE','R²','SMAPE'],rows)
para(doc,f'随机森林是当前数据和特征设计下的最优主模型，测试集 RMSE 为 {metrics["random_forest"]["test"]["RMSE"]:.3f} mg/Nm³，R² 为 {metrics["random_forest"]["test"]["R2"]:.4f}。持续性基线的误差也较小，说明 10 s 聚合后的过程具有明显惯性；因此模型适合作为短时预测和控制辅助，不应绕过现场联锁直接替代闭环保护。')
if (ART/'prediction_test.png').exists(): doc.add_picture(str(ART/'prediction_test.png'),width=Inches(6.7)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
heading(doc,'5 变量重要性与工艺分析',1)
para(doc,'随机森林置换重要性表明，尾部 NOx 历史值、A/B 侧出口 NOx 的历史值和历史脱硝效率是主要信息源，符合 SCR 反应器大时延和过程惯性。入口 NOx、喷氨量、烟气温度与负荷提供扰动和操纵信息。')
table(doc,['特征','置换重要性'],[[r.feature,f'{r.importance:.4f}'] for r in imp.itertuples()])
if (ART/'feature_importance.png').exists(): doc.add_picture(str(ART/'feature_importance.png'),width=Inches(6.7)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_page_break()
heading(doc,'6 约束喷氨控制设计',1)
para(doc,'使用随机森林作为数据驱动数字孪生模型，在当前 A/B 侧氨流量的 0.80–1.20 倍范围内进行网格搜索。约束条件为预测尾部 NOx 不高于当前设定值加 2 mg/Nm³，目标是在满足约束的候选解中最小化总氨流量。')
table(doc,['项目','当前值','离线推荐'],[['负荷',f'{control["current"]["load_mw"]:.1f} MW',f'{control["current"]["load_mw"]:.1f} MW'],['A侧氨流量',f'{control["current"]["nh3_a"]:.2f} kg/h',f'{control["recommendation"]["nh3_a"]:.2f} kg/h'],['B侧氨流量',f'{control["current"]["nh3_b"]:.2f} kg/h',f'{control["recommendation"]["nh3_b"]:.2f} kg/h'],['预测尾部 NOx',f'{control["current"]["predicted_tail_nox"]:.2f} mg/Nm³',f'{control["recommendation"]["pred_tail_nox"]:.2f} mg/Nm³'],['设定值',f'{control["current"]["setpoint"]:.2f} mg/Nm³',f'{control["recommendation"]["target_setpoint"]:.2f} mg/Nm³']])
para(doc,'上述结果是基于历史数据模型的离线建议，不能直接作为现场控制指令。实际闭环实施应增加氨逃逸约束、阀门变化率、最小稳定喷氨量、催化剂活性修正、测量延迟和联锁状态，并在 Simulink 或实时仿真环境中完成抗扰和阶跃测试。')
heading(doc,'7 DeepSeek API 与网页系统',1)
para(doc,'DeepSeek API 仅用于根据统计摘要、指标、变量重要性和控制推荐生成可读解释、异常工况说明和研究报告文字。原始逐秒数据清洗、特征计算、模型训练、指标评价和控制搜索全部由本地 Python 完成。API 密钥仅保存在后端环境变量 DEEPSEEK_API_KEY 中，网页前端不接触密钥。')
para(doc,'网页系统使用 FastAPI 提供 /api/summary、/api/predict、/api/control 和 /api/deepseek 接口，前端展示 NOx 趋势、A/B 侧过程量、测试集预测、变量重要性、控制推荐和自然语言分析。')
heading(doc,'8 结论与后续工作',1)
para(doc,'本设计完成了任务书要求的数据分析、数据对齐、预处理、传统机器学习、神经网络建模、精度评价、工艺变量分析和控制辅助设计。随机森林对稳定工况尾部 NOx 具有较高预测精度，GRU 提供了独立的时序建模通道，融合结果可作为保守的多模型参考。')
para(doc,'数据仍只有单台机组约 30 h 记录，且包含分析仪上限值和启停/切换段。后续应补充多日期、多负荷、多煤种数据以及氨逃逸实测值，并在 Simulink 或实时仿真平台进行闭环控制、扰动恢复和安全边界验证。')
heading(doc,'参考资料',1)
para(doc,'[1] 2025 任务书：脱硝过程的建模与控制。\n[2] 源数据：靖海电厂 1 号 600 MW 机组脱硝相关数据（0 MW–600 MW）。\n[3] 课程设计中使用的本地数据处理、随机森林、GRU 和 FastAPI 实现。')
# footer page numbers
for section in doc.sections:
 footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; add_text(footer,'SCR 脱硝数据驱动建模与控制课程设计',9,color=(120,120,120))
out=ROOT/'脱硝过程建模与控制报告.docx'; doc.save(out); print(out)
