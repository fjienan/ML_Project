# Own Problem: Node-Coded Prediction

本目录实现课程项目第四部分自选问题：

> 能否利用乳腺癌蛋白质组表达预测淋巴结状态 `Node-Coded`？

运行方式：

```bash
python3 -m pip install --user -r requirements.txt
python3 code/own_problem/node_prediction.py
```

脚本会自动寻找数据文件，优先读取 `code/preprocessing/`，如果该目录为空，则读取 `code/` 根目录。

模型比较三种方法：

- `LogisticRegression`: 可解释线性基线。
- `RandomForest`: bagging-family 集成学习基线。
- `DAE_MLP`: Denoising AutoEncoder + MLP 深度表示学习模型。

主要输出：

- `code/own_problem/results/cv_metrics_summary.csv`
- `code/own_problem/results/cv_metrics_by_fold.csv`
- `code/own_problem/results/oof_predictions.csv`
- `code/own_problem/results/top_protein_importance.csv`
- `article/figures/own_problem/*.png`

注意：该任务样本数只有 77 个，报告重点应放在多模型比较和小样本高维任务的局限分析，而不是强行追求高分。
