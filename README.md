# ML_Project 📁

南方科技大学 -- 大数据科学导论：课程项目 II

## 📌 项目简介

本项目（Project II，截止日期：2026 年 6 月 21 日）基于美国国家癌症研究所（NCI/NIH）临床蛋白质组肿瘤分析联盟（CPTAC）发布的**乳腺癌蛋白质组数据**，运用机器学习技术（聚类与分类）对该数据进行深入分析。

数据集包含三个 CSV 文件：

- **`77_cancer_proteomes_CPTAC_itraq.csv`** -- 77 个乳腺癌样本及 3 个健康样本的蛋白质表达量（log2 iTRAQ 比值），覆盖 12,000 余种蛋白质。
- **`clinical_data_breast_cancer.csv`** -- 105 个乳腺癌样本的临床检查结果（ID、肿瘤分类等）。
- **`PAM50_proteins.csv`** -- PAM50 分类系统所用的基因/蛋白质列表。

数据来源：[Nature, 2016](http://www.nature.com/nature/journal/v534/n7605/full/nature18003.html)

## ✅ 任务清单

1. **数据预处理** -- 缺失值检测、异常值处理、标准化、训练/测试集划分。
2. **聚类分析** -- 基于蛋白质表达数据进行 K-means 聚类，与 PAM50 mRNA 分类结果对比。
3. **分类分析** -- 以 PAM50 mRNA 为标签进行监督学习（43 个训练样本 / 34 个测试样本）。
4. **自选问题** -- 设计一个额外的机器学习问题并加以分析。
5. **结论**

## 🗂️ 目录结构

```
questions/          -- 项目说明 PDF 及原始数据文件
article/            -- 课程论文 LaTeX 源码
  main.tex          -- 主入口文件
  sections/         -- 各章节 .tex 文件
  figures/          -- 图片存放目录
  Makefile          -- 编译工具
  .gitignore
```

---

## 📄 如何编辑与生成 PDF

本项目使用 **LaTeX** 撰写论文，推荐以下两种使用方式：

### 💻 方式一：命令行编译（适用于任何编辑器）

```bash
cd article
make          # 编译 main.pdf
make view     # 编译并自动打开 PDF
make clean    # 清理编译产物（.aux、.log、.pdf 等）
```

> **提示**：LaTeX 通常需要编译**两次**才能生成正确的目录引用（ToC）和交叉引用，`Makefile` 已自动处理了这一点。若 PDF 中参考文献或交叉引用显示 `?`，再执行一次 `make` 即可。

### 📝 main.tex 文件结构说明

`article/main.tex` 是整篇论文的入口文件，采用"主文件 + 分章节"的模块化结构。文件各部分含义如下：

**文件头（Package 与格式设置）**

| 行号 | 内容 | 说明 |
|------|------|------|
| 1 | `\documentclass[11pt,a4paper]{article}` | 文档类型：11pt 字号、A4 纸、单栏 article |
| 4–18 | `\usepackage{...}` | 加载数学公式、绘图、代码高亮等宏包 |
| 27–32 | `\geometry{...}` | 页边距：上下左右各 2.5 cm |
| 35–39 | `\pagestyle{fancy}` | 页眉页脚：页眉显示课程名，页脚居中显示页码 |

**正文区**

| 行号 | 内容 | 说明 |
|------|------|------|
| 62–70 | `\begin{document} ... \end{center}` | 封面：标题、学校、课程、截止日期 |
| 76–83 | `\input{sections/...}` | 依次引入七个章节文件 |

**写作提示**

- **每个章节**对应 `sections/` 下的一个 `.tex` 文件（如 `sections/1_introduction.tex`），在对应文件中直接写内容即可，无需再写 `\section{}`，因为文件内容会被 `\input` 原地展开。
- **图表**统一放在 `article/figures/` 目录下，在 `.tex` 中用 `\includegraphics{figures/文件名}` 引用。
- **代码高亮**可直接用 `\begin{lstlisting}` 块（已预置 Python 语法配色），或粘贴 Jupyter Notebook 导出的图片。
- **数学公式**用 `equation` / `align` 环境，公式会自动编号。
- **交叉引用**：在章节中用 `\label{sec:xxx}` 打标签，正文中用 `\ref{sec:xxx}` 引用，编译两次后编号自动更新。

### ⚙️ 依赖环境

**Python（数据分析）**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

**LaTeX 发行版（任选其一）**
- Linux / macOS：安装 [TeX Live](https://www.tug.org/texlive/)
- Windows：安装 [MiKTeX](https://miktex.org/) 或 TeX Live
- macOS：也可使用 [MacTeX](https://www.tug.org/mactex/)

验证 LaTeX 是否安装成功：
```bash
pdflatex --version
```
