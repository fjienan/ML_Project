"""
第三题：随机森林分类（强化防过拟合 + 网格搜索优化版）
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import tree

# ===================== 自动创建保存路径 =====================
ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
save_dir = ROOT / "article" / "figures" / "classification"
save_dir.mkdir(parents=True, exist_ok=True)

# ===================== 1. 读取数据 =====================
train_set = pd.read_csv(CODE_DIR / "train_set.csv", index_col=0)
test_set = pd.read_csv(CODE_DIR / "test_set.csv", index_col=0)

# ===================== 2. 特征与标签 =====================
X_train = train_set.drop(columns=['PAM50 mRNA'])
y_train = train_set['PAM50 mRNA']
X_test = test_set.drop(columns=['PAM50 mRNA'])
y_test = test_set['PAM50 mRNA']

# ===================== 3. 优化版参数网格 =====================
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# ===================== 4. 输出最优参数 =====================
print("=" * 60)
print("✅ 最优参数：", grid_search.best_params_)
print("=" * 60)

# ===================== 5. 预测与评估 =====================
y_pred = best_rf.predict(X_test)
train_acc = best_rf.score(X_train, y_train)
test_acc = best_rf.score(X_test, y_test)

print("\n训练集准确率：", round(train_acc, 3))
print("测试集准确率：", round(test_acc, 3))
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ===================== 6. 特征重要性 =====================
imp_df = pd.DataFrame({
    "Protein_ID": X_train.columns,
    "Importance": best_rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n🔥 Top 10 关键蛋白质：")
print(imp_df.head(10))

# ==============================================
# 可视化1：Top10 蛋白质重要性（自动保存）
# ==============================================
plt.figure(figsize=(10, 6))
plt.barh(imp_df["Protein_ID"][:10][::-1], imp_df["Importance"][:10][::-1], color='#1f77b4')
plt.xlabel("Importance Score")
plt.title("Top 10 Important Proteins (Optimized Random Forest)")
plt.tight_layout()
plt.savefig(save_dir / "rf_top10_proteins.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================
# 可视化2：训练 vs 测试准确率（自动保存）
# ==============================================
plt.figure(figsize=(6, 5))
models = ['Train Accuracy', 'Test Accuracy']
accs = [train_acc, test_acc]
colors = ['#2ca02c', '#ff7f0e']
plt.bar(models, accs, color=colors)
plt.ylim(0, 1.1)
plt.title('Random Forest Train vs Test Accuracy')
plt.ylabel('Accuracy')
for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(save_dir / "rf_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================
# 可视化3：混淆矩阵（自动保存）
# ==============================================
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(save_dir / "rf_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================
# 可视化4：单棵决策树结构（自动保存）
# ==============================================
plt.figure(figsize=(18, 12))
tree.plot_tree(
    best_rf.estimators_[0],
    feature_names=X_train.columns,
    class_names=labels,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title('Single Decision Tree from Optimized Random Forest')
plt.tight_layout()
plt.savefig(save_dir / "rf_tree_structure.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ 所有随机森林图片已保存到：{save_dir}")
