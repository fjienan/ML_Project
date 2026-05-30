# Code Organization

本目录保存课程项目的数据、实验脚本和实验结果。CSV 数据保留在 `code/` 根目录，脚本按任务拆分到子目录。

## Structure

```text
code/
├── preprocessing/
│   ├── 1_data_processing.py
│   └── 2_data_exploration.py
├── clustering/
│   ├── dec_clustering.py
│   └── results/
├── classification/
│   ├── 3_random_forest.py
│   ├── 3_deep_forest.py
│   ├── 3_naive_bayes.py
│   ├── 3_tan_baseline.py
│   ├── 3_nb_tan_fusion.py
│   └── 3_model_comparison.py
├── own_problem/
│   ├── node_prediction.py
│   └── results/
└── *.csv
```

## Run Order

从项目根目录运行：

```bash
python3 code/preprocessing/1_data_processing.py
python3 code/preprocessing/2_data_exploration.py
python3 code/clustering/dec_clustering.py
python3 code/classification/3_random_forest.py
python3 code/classification/3_deep_forest.py
python3 code/classification/3_naive_bayes.py
python3 code/classification/3_nb_tan_fusion.py
python3 code/classification/3_model_comparison.py
python3 code/own_problem/node_prediction.py
```

## Outputs

- 训练/测试集：`code/train_set.csv`、`code/test_set.csv`
- 预处理与探索图：`article/figures/data_preprocessing/`
- 聚类图与结果：`article/figures/clustering/`、`code/clustering/results/`
- 分类图：`article/figures/classification/`
- 自选问题图与结果：`article/figures/own_problem/`、`code/own_problem/results/`
