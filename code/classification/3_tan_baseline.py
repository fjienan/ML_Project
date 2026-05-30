"""
TAN 树增强朴素贝叶斯（最高级版本）
特征之间形成树结构，不是朴素独立
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer

# 数据
ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
train = pd.read_csv(CODE_DIR / "train_set.csv", index_col=0)
test = pd.read_csv(CODE_DIR / "test_set.csv", index_col=0)

X_train = train.drop("PAM50 mRNA", axis=1).values
y_train = train["PAM50 mRNA"].values
X_test = test.drop("PAM50 mRNA", axis=1).values
y_test = test["PAM50 mRNA"].values

dis = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
X_train_d = dis.fit_transform(X_train)
X_test_d = dis.transform(X_test)

classes = np.unique(y_train)
n_features = X_train.shape[1]

# 简化 TAN 结构：每个特征 i 父节点为 i-1（树结构）
parents = {i: i-1 for i in range(1, n_features)}
parents[0] = None

# 概率表
prior = {c: np.mean(y_train == c) for c in classes}
cond = {}

for c in classes:
    idx = y_train == c
    Xc = X_train_d[idx]
    cond[c] = {}
    for i in range(n_features):
        p = parents[i]
        for v in np.unique(X_train_d[:, i]):
            if p is None:
                cnt = np.sum(Xc[:, i] == v)
                cond[c][(i, v)] = (cnt + 1) / (len(Xc) + 5)
            else:
                for vp in np.unique(X_train_d[:, p]):
                    cnt = np.sum((Xc[:, i] == v) & (Xc[:, p] == vp))
                    cond[c][(i, v, vp)] = (cnt + 1) / (len(Xc) + 5)

# TAN 预测
def predict_tan(x):
    scores = {}
    for c in classes:
        s = np.log(prior[c])
        for i in range(n_features):
            p = parents[i]
            if p is None:
                s += np.log(cond[c].get((i, x[i]), 1e-6))
            else:
                s += np.log(cond[c].get((i, x[i], x[p]), 1e-6))
        scores[c] = s
    return max(scores, key=scores.get)

y_pred = [predict_tan(x) for x in X_test_d]
train_pred = [predict_tan(x) for x in X_train_d]

print("===== TAN 树增强朴素贝叶斯 =====")
print(f"训练准确率：{accuracy_score(y_train, train_pred):.3f}")
print(f"测试准确率：{accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
