# Clustering: DEC for Project Task 2

本目录用于课程项目任务二：使用蛋白质表达数据进行无监督聚类，并与 `PAM50 mRNA` 分型对比。

运行方式：

```bash
python3 -m pip install --user -r requirements.txt
python3 code/clustering/dec_clustering.py
```

脚本会自动寻找数据文件，优先读取 `code/preprocessing/`，如果该目录为空，则读取 `code/` 根目录中的课程 CSV 文件。

主要输出：

- `code/clustering/results/kmeans_by_k_metrics.csv`
- `code/clustering/results/method_metrics_summary.csv`
- `code/clustering/results/cluster_assignments.csv`
- `code/clustering/results/pam50_dec_crosstab.csv`
- `article/figures/clustering/*.png`

说明：

- `PAM50 mRNA` 只用于外部评价，不参与聚类训练。
- 主聚类数为 `k=3`，同时保留 `k=4` 与 PAM50 四分类作补充比较。
- 聚类代码目录统一命名为 `code/clustering/`。
