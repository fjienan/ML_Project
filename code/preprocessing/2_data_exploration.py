# ===================== 乳腺癌数据 分开可视化（每张图独立） =====================
import os
import pandas as pd
import re
import numpy as np
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
FIGURE_DIR = ROOT / "article" / "figures" / "data_preprocessing"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 读取数据 =====================
dataset_path = CODE_DIR / "77_cancer_proteomes_CPTAC_itraq.csv"
clinical_info = CODE_DIR / "clinical_data_breast_cancer.csv"
pam50_proteins = CODE_DIR / "PAM50_proteins.csv"

data_raw = pd.read_csv(dataset_path, header=0, index_col=0)
clinical = pd.read_csv(clinical_info, header=0, index_col=0)
pam50 = pd.read_csv(pam50_proteins, header=0)

# ==============================================================================
# ======================== 【一】数据处理前 —— 每张图独立 ========================
# ==============================================================================

# 1. 原始数据缺失值分布（患者缺失某蛋白质显示红色，不缺失就不显色）
plt.figure(figsize=(10, 6))
sns.heatmap(data_raw.isnull(), cbar=False, cmap="Reds", yticklabels=False)
plt.title("Missing Value Distribution of Raw Data", fontsize=14)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "1_missing_value.png", dpi=300)
plt.close()

# 2. 原始蛋白质表达值分布
expr_raw = data_raw.drop(columns=["gene_symbol", "gene_name"], errors="ignore")
expr_flat = expr_raw.values.flatten()
expr_flat = expr_flat[~np.isnan(expr_flat)]

plt.figure(figsize=(10, 6))
plt.hist(expr_flat, bins=50, color="skyblue", alpha=0.7)
plt.title("Distribution of Raw Protein Expression Values", fontsize=14)
plt.xlabel("Expression Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "2_distribution_raw_pro_value.png", dpi=300)
plt.close()

# 3. 原始临床 PAM50 亚型分布
plt.figure(figsize=(10, 6))
clinical["PAM50 mRNA"].value_counts().plot(kind="bar", color="orange", alpha=0.7)
plt.title("Distribution of Raw Clinical PAM50 Subtypes", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "3_distribution_raw_pam50.png", dpi=300)
plt.close()
#PAM50 是乳腺癌分子分型（LumA、LumB、Basal、HER2 等）
#这张图展示各类别样本数量是否均衡
#能看出数据集中哪一类患者最多，为后续分类模型提供依据

# ==============================================================================
# ======================== 【二】执行你的数据处理流程 ========================
# ==============================================================================
data = data_raw.copy()
data.drop(['gene_symbol', 'gene_name'], axis=1, inplace=True)

data.rename(
    columns=lambda x: "TCGA-%s" % re.split(r'[_|.]', x)[0] if "TCGA" in x else x,
    inplace=True
)
data = data.loc[:, ~data.columns.duplicated()]
data = data.transpose()
data = data[data.index.str.contains("TCGA")]
final_data = data.merge(clinical, left_index=True, right_index=True)

pam50_ids = pam50['RefSeqProteinID'].tolist()
feature_cols = [x for x in final_data.columns if x in pam50_ids]
X = final_data[feature_cols]
y = final_data['PAM50 mRNA']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
final_df = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
final_df['PAM50 mRNA'] = y.values

train_set, test_set = train_test_split(
    final_df, test_size=34/77, random_state=42, stratify=final_df['PAM50 mRNA']
)

# ==============================================================================
# ======================== 【三】数据处理后 —— 每张图独立 ========================
# ==============================================================================

# 2. 处理后表达值分布
plt.figure(figsize=(10, 6))
plt.hist(final_df[feature_cols].values.flatten(), bins=50, color="lightgreen", alpha=0.7)
plt.title("Distribution of Processed Protein Expression Values", fontsize=14)
plt.xlabel("Expression Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "4_distribution_processed_pro_value.png", dpi=300)
plt.close()

# 3. 训练集 / 测试集 PAM50 分层分布对比
plt.figure(figsize=(10, 6))
train_pam = train_set['PAM50 mRNA'].value_counts().sort_index()
test_pam = test_set['PAM50 mRNA'].value_counts().sort_index()
x = np.arange(len(train_pam))
width = 0.35

plt.bar(x - width/2, train_pam, width, label='Training Set', color='dodgerblue')
plt.bar(x + width/2, test_pam, width, label='Testing Set', color='coral')
plt.title('Distribution of PAM50 Subtypes in Training vs Testing Sets', fontsize=14)
plt.xticks(x, train_pam.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "5_training_testing_pam50_comparison.png", dpi=300)
plt.close()

# ===================== 保存文件 =====================
train_set.to_csv(CODE_DIR / "train_set.csv", encoding="utf-8-sig")
test_set.to_csv(CODE_DIR / "test_set.csv", encoding="utf-8-sig")

print(f"\n✅ 所有图片已保存到：{FIGURE_DIR}")
print(f"✅ 训练集/测试集已生成到：{CODE_DIR}")
