import pandas as pd
import numpy as np
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    cohen_kappa_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import pearsonr
from itertools import cycle

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
FIGURE_DIR = ROOT / "article" / "figures" / "classification"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 读取数据 =====================
train = pd.read_csv(CODE_DIR / "train_set.csv", index_col=0)
test = pd.read_csv(CODE_DIR / "test_set.csv", index_col=0)

feature_names = train.drop("PAM50 mRNA", axis=1).columns
X_train = train.drop("PAM50 mRNA", axis=1).values
y_train = train["PAM50 mRNA"].values
X_test = test.drop("PAM50 mRNA", axis=1).values
y_test = test["PAM50 mRNA"].values

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
n_classes = len(le.classes_)

# 标签二值化（用于多分类ROC）
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== TAN 模型 =====================
class TANClassifier:
    def __init__(self):
        self.classes_ = None
        self.n_classes = 0
        self.n_features = 0
        self.theta = None
        self.var = None
        self.prior = None
        self.parent = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        self.theta = np.zeros((self.n_classes, self.n_features))
        self.var = np.zeros((self.n_classes, self.n_features))
        for c_idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta[c_idx] = np.mean(X_c, axis=0)
            self.var[c_idx] = np.var(X_c, axis=0)

        self.var = np.clip(self.var, 1e-8, np.percentile(self.var, 90))
        self.prior = np.log(np.bincount(y) / len(y) + 1e-10)
        dist_matrix = 1 - np.abs(np.corrcoef(X.T))
        mst = minimum_spanning_tree(csr_matrix(dist_matrix)).toarray()
        self.parent = np.full(self.n_features, -1)
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                if mst[i,j] > 0 and self.parent[j] == -1:
                    self.parent[j] = i
        return self

    def _log_prob(self, x):
        log_probs = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            lp = 0
            for f in range(self.n_features):
                mu = self.theta[c,f]
                var = self.var[c,f]
                lp += -0.5*np.log(2*np.pi*var) - ((x[f]-mu)**2)/(2*var)
            log_probs[c] = lp + self.prior[c]
        return log_probs

    def predict(self, X):
        return np.array([self.classes_[np.argmax(self._log_prob(x))] for x in X])
    
    def predict_log_proba(self, X):
        return np.array([self._log_prob(x) for x in X])

# ===================== 训练模型 =====================
tan = TANClassifier()
tan.fit(X_train, y_train)
nb = GaussianNB()
nb.fit(X_train, y_train)

# ===================== 融合预测函数（返回概率/对数概率） =====================
def predict_ensemble_log_prob(X, alpha):
    # 返回 (n_samples, n_classes) 对数概率
    nb_log = nb.predict_log_proba(X)
    tan_log = tan.predict_log_proba(X)
    fused = alpha * nb_log + (1 - alpha) * tan_log
    return fused

# ===================== 计算各模型指标（含AUC） =====================
alphas = [1.0, 0.524, 0.0]
model_names = ["纯NB", "融合α=0.524", "纯TAN"]
plot_names = ["Gaussian NB", "Fusion alpha=0.524", "TAN"]
results = []

for a, name in zip(alphas, model_names):
    # 对数概率
    log_prob = predict_ensemble_log_prob(X_test, a)
    # 预测标签
    y_pred = np.argmax(log_prob, axis=1)
    # 概率（用于AUC）
    prob = np.exp(log_prob)
    
    # 指标
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    macro_auc = roc_auc_score(y_test_bin, prob, average='macro')
    micro_auc = roc_auc_score(y_test_bin, prob, average='micro')
    
    results.append({
        "模型": name,
        "准确率": acc,
        "Kappa": kappa,
        "Macro-AUC": macro_auc,
        "Micro-AUC": micro_auc
    })

# 打印对比表
print("\n===== 模型性能对比（含AUC）=====")
for res in results:
    print(f"{res['模型']:10} | Acc: {res['准确率']:.4f} | Kappa: {res['Kappa']:.4f} | Macro-AUC: {res['Macro-AUC']:.4f} | Micro-AUC: {res['Micro-AUC']:.4f}")

# ===================== 绘制ROC曲线（3模型同图对比） =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
linestyles = ['--', '-', ':']

for idx, (a, name) in enumerate(zip(alphas, plot_names)):
    prob = np.exp(predict_ensemble_log_prob(X_test, a))
    # 微平均ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color=colors[idx], linestyle=linestyles[idx],
             lw=2, label=f'{name} (Micro-AUC={auc_micro:.4f})')

# 随机线
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random baseline')
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('Gaussian NB vs Fusion vs TAN ROC Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 绘制各模型AUC柱状图 =====================
plt.figure(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.35

macro_aucs = [res['Macro-AUC'] for res in results]
micro_aucs = [res['Micro-AUC'] for res in results]

plt.bar(x - width/2, macro_aucs, width, label='Macro-AUC', color='#1f77b4')
plt.bar(x + width/2, micro_aucs, width, label='Micro-AUC', color='#ff7f0e')

plt.xlabel('Model')
plt.ylabel('AUC')
plt.title('Macro-AUC and Micro-AUC Comparison')
plt.xticks(x, plot_names)
plt.ylim(0.8, 0.95)
plt.legend()
# 加数值标签
for i, v in enumerate(macro_aucs):
    plt.text(i - width/2, v + 0.005, f'{v:.4f}', ha='center')
for i, v in enumerate(micro_aucs):
    plt.text(i + width/2, v + 0.005, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ===================== 融合模型 Top10 特征重要性 =====================
nb_score = np.std(nb.theta_, axis=0) / (np.mean(nb.var_, axis=0) + 1e-8)
tan_score = np.std(tan.theta, axis=0) / (np.mean(tan.var, axis=0) + 1e-8)
fusion_score = 0.524 * nb_score + (1 - 0.524) * tan_score
top10 = pd.Series(fusion_score, index=feature_names).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top10.iloc[::-1].plot(kind='barh', color='#9467bd', ax=ax)
ax.set_xlabel('Fusion Discrimination Score')
ax.set_title('Top 10 Important Protein Features (NB + TAN Fusion)')
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / 'nb_tan_top10_features.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n✅ 已生成 ROC、AUC 和融合 Top10 特征图到：{FIGURE_DIR}")
