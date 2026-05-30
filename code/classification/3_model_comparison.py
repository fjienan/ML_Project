import numpy as np
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 数据（和论文表格保持一致）
models = ['Deep Forest', 'Random Forest', 'Gaussian NB', 'Fusion']
test_acc = [0.794, 0.765, 0.824, 0.882]
fit_gap = [0.206, 0.235, 0.106, 0.048]

# 保存路径
base_dir = Path(__file__).resolve().parents[2]
save_dir = base_dir / "article" / "figures" / "classification"
os.makedirs(save_dir, exist_ok=True)

# ===================== 图1：Test Accuracy =====================
plt.figure(figsize=(8, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(models, test_acc, color=colors)
plt.ylim(0.6, 0.93)
plt.title('Model Test Accuracy Comparison', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=12)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(save_dir / "model_test_acc.png", dpi=300, bbox_inches='tight')
plt.close()

# ===================== 图2：Fit Gap =====================
plt.figure(figsize=(8, 5))
bars = plt.bar(models, fit_gap, color=colors)
plt.ylim(0, 0.35)
plt.title('Model Train-Test Fit Gap Comparison', fontsize=14)
plt.ylabel('Fit Gap (Train - Test Accuracy)', fontsize=12)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(save_dir / "model_fit_gap.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All figures saved to {save_dir}")
