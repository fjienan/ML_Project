# ML Project 🎯

南方科技大学"大数据科学导论"课程项目 II。项目基于 CPTAC 乳腺癌蛋白质组数据，完成数据预处理、探索性分析、聚类分析、PAM50 分型分类，以及一个自选临床预测问题。

## 📁 目录结构

```
.
├── article/                  # LaTeX 报告源码与报告图表
│   ├── main.tex              # 报告入口
│   ├── sections/             # 各章节 tex 文件
│   ├── figures/              # 报告中引用的图表
│   └── Makefile              # PDF 编译命令
├── code/                     # 数据、实验脚本与实验结果
│   ├── preprocessing/        # 预处理与数据探索
│   ├── classification/        # PAM50 分类实验
│   ├── clustering/           # K-means 与 DEC 聚类实验
│   ├── own_problem/          # Node-Coded 自选问题实验
│   └── *.csv                 # 原始数据副本与处理后数据
├── questions/                # 课程说明、原始数据压缩包与原始数据
├── requirements.txt          # Python 依赖
└── README.md
```

## 📊 项目成果

### 论文报告内容

| 章节 | 内容描述 |
|------|----------|
| 数据预处理 | 缺失值处理（KNN/中位数填补）、标准化（Z-score/Log）、特征筛选 |
| 探索性分析 | 样本相关性热力图、蛋白质表达分布、主成分分析（PCA）可视化 |
| 聚类分析 | K-means++ 聚类与 DEC（Deep Embedded Clustering）深度聚类 |
| PAM50 分类 | Random Forest、Deep Forest、Naive Bayes、NB-TAN Fusion 四种分类器对比 |
| 自选问题 | 基于 Node-Coded 标签的临床预测（Logistic Regression / Random Forest / DAE+MLP） |

### 核心结果

- ✅ 预处理后数据质量提升，缺失值填补效果显著
- ✅ DEC 聚类在 PAM50 mRNA 标签下 ARI 达到 **0.38**（未调参基线）
- ✅ Random Forest 在 PAM50 分型分类中 F1-score 达 **0.76**
- ✅ DAE+MLP 在 Node-Coded 预测中 AUC 达 **0.84**

## 📂 数据文件

主要数据位于 `code/` 和 `questions/Course_Project_II/`：

- `77_cancer_proteomes_CPTAC_itraq.csv`：乳腺癌样本蛋白质表达矩阵。
- `clinical_data_breast_cancer.csv`：临床信息与 PAM50、Node-Coded 等标签。
- `PAM50_proteins.csv`：PAM50 分型相关蛋白列表。

## ⚙️ 环境配置

Python 依赖安装：

```bash
pip install -r requirements.txt
```

报告编译需要 XeLaTeX/TeX Live 或兼容 LaTeX 发行版。

## 🔬 推荐复现顺序

以下命令均从项目根目录运行：

```bash
python3 code/preprocessing/1_data_processing.py
python3 code/preprocessing/2_data_exploration.py
python3 code/clustering/dec_clustering.py
python3 code/own_problem/node_prediction.py
python3 code/classification/3_random_forest.py
python3 code/classification/3_deep_forest.py
python3 code/classification/3_naive_bayes.py
python3 code/classification/3_nb_tan_fusion.py
python3 code/classification/3_model_comparison.py
```

生成或更新 PDF：

```bash
cd article
make -B
```

## 📤 输出位置

- 预处理后的训练/测试集：`code/train_set.csv`、`code/test_set.csv`
- 聚类结果表：`code/clustering/results/`
- 自选问题结果表：`code/own_problem/results/`
- 报告图表：`article/figures/`
- 最终报告：`article/main.pdf`

## 📝 说明

- `PAM50 mRNA` 标签仅用于聚类结果的外部评价，不参与聚类训练。
- 预处理流程保持中位数填补和固定随机种子的分层训练/测试拆分。
- 自选问题使用 `Node-Coded` 标签，比较 Logistic Regression、Random Forest 与 Denoising AutoEncoder + MLP。
