"""
朴素贝叶斯（小样本专用｜几乎不过拟合）
纯贝叶斯逻辑的Top10蛋白质筛选（无随机森林依赖）
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== 1. 读取数据 =====================
ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
train = pd.read_csv(CODE_DIR / "train_set.csv", index_col=0)
test = pd.read_csv(CODE_DIR / "test_set.csv", index_col=0)

X_train = train.drop("PAM50 mRNA", axis=1)
y_train = train["PAM50 mRNA"]
X_test = test.drop("PAM50 mRNA", axis=1)
y_test = test["PAM50 mRNA"]

# ===================== 2. 创建保存图片的路径 =====================
save_dir = ROOT / "article" / "figures" / "classification"
save_dir.mkdir(parents=True, exist_ok=True)

# ===================== 3. 朴素贝叶斯模型 =====================
model = GaussianNB()
model.fit(X_train, y_train)

# 计算特征间平均相关系数
corr = np.corrcoef(X_train.T)
corr_abs = np.abs(corr)
np.fill_diagonal(corr_abs, 0)
avg_corr = np.mean(corr_abs)
print(f"特征间平均相关系数绝对值：{avg_corr:.3f}")

# 预测
y_pred = model.predict(X_test)
train_acc = model.score(X_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

print("【朴素贝叶斯】")
print(f"训练准确率：{train_acc:.3f}")
print(f"测试准确率：{test_acc:.3f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ===================== 核心修改：纯贝叶斯的Top10特征筛选 =====================
# 步骤1：提取GaussianNB的核心统计量（每个类别下的特征均值+方差）
class_labels = model.classes_  # PAM50亚型标签（如LumA、Basal等）
feature_means = model.theta_    # 每个类别下各特征的均值 (n_classes, n_features)
feature_vars = model.var_      # 每个类别下各特征的方差 (n_classes, n_features)

# 步骤2：计算每个特征的“贝叶斯区分度”
# 逻辑：不同类别下特征均值的变异系数（均值标准差 / 整体均值）→ 越大说明该特征越能区分类别
feature_names = X_train.columns
bayes_importance = []

for i, feat in enumerate(feature_names):
    # 该特征在所有类别下的均值
    feat_means = feature_means[:, i]
    # 该特征在所有类别下的方差（取平均，代表整体波动）
    avg_var = np.mean(feature_vars[:, i]) + 1e-8  # 加小值避免除0
    # 区分度得分 = 均值的标准差 / 平均方差（得分越高，特征越关键）
    score = np.std(feat_means) / avg_var
    bayes_importance.append(score)

# 步骤3：筛选Top10特征
bayes_importance_series = pd.Series(bayes_importance, index=feature_names)
top10_features = bayes_importance_series.sort_values(ascending=False).head(10)

# 输出Top10列表（纯贝叶斯逻辑）
print("\n✅ 纯贝叶斯逻辑的Top10关键蛋白质特征：")
for idx, (feat, score) in enumerate(top10_features.items(), 1):
    print(f"{idx}. {feat} (贝叶斯区分度得分：{score:.4f})")

# ===================== Top10可视化（仅改标题，样式不变） =====================
plt.figure(figsize=(10, 6))
top10_features.plot(kind='barh', color='#2ecc71')
plt.xlabel('Bayesian Discrimination Score')  # 改为贝叶斯区分度得分
plt.ylabel('Protein Feature Name')
plt.title('Top 10 Important Protein Features (Gaussian Naive Bayes)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(save_dir / "nb_top10_features_bayesian.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================
# 原有可视化逻辑（完全保留，无任何修改）
# ==============================================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(save_dir / "nb_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(6, 5))
plt.bar(['Train Acc', 'Test Acc'], [train_acc, test_acc], color=['#1f77b4', '#ff7f0e'])
plt.ylim(0, 1.05)
plt.title('Gaussian NB Train vs Test Accuracy')
plt.ylabel('Accuracy')
for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(save_dir / "nb_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
sample_features = X_train.columns[:50]
sns.heatmap(X_train[sample_features].corr(), cmap='coolwarm', vmax=1, vmin=-1)
plt.title(f'Protein Feature Correlation (Avg Abs: {avg_corr:.3f})')
plt.tight_layout()
plt.savefig(save_dir / "nb_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 所有图片（含贝叶斯Top10特征图）已保存到：{save_dir}")
