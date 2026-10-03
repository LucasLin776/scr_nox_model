# SCR 脱硝过程数据驱动建模与控制

本目录包含靖海电厂 1 号 600 MW 机组脱硝实测数据的清洗、建模、控制优化和网页可视化结果。

## 运行网页

```bash
cd /Users/shuoqinlin/scr_nox_model
/Users/shuoqinlin/miniconda3/envs/tool-agent/bin/python run_server.py
```

浏览器打开 `http://127.0.0.1:8000/`。

## 重新训练

```bash
/Users/shuoqinlin/miniconda3/envs/tool-agent/bin/python inspect_data.py
/Users/shuoqinlin/miniconda3/envs/tool-agent/bin/python train_model.py
/Users/shuoqinlin/miniconda3/envs/tool-agent/bin/python finalize_outputs.py
```

## DeepSeek API

复制 `.env.example` 中的配置到后端进程环境变量。API 调用由 `/api/deepseek` 完成，发送的是指标、特征重要性、数据质量统计和控制推荐，不发送原始逐秒数据。

## 主要结果

- `report.md`：课程设计报告草稿
- `artifacts/metrics.json`：模型指标
- `artifacts/control_recommendation.json`：当前末端工况的离线喷氨推荐
- `artifacts/prediction_test.png`：测试集预测图
- `artifacts/feature_importance.png`：变量重要性图
- `artifacts/*.joblib`、`artifacts/gru.pt`：模型文件
- `clean_10s.csv`：清洗后 10 s 聚合数据
