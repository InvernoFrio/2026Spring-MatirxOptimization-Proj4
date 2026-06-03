# Pegasos SVM 项目实验报告

## 1. 项目目标

本项目围绕 SVM 在不同规模、不同数据形态下的训练方法展开：

- 在线性可分或近似线性可分任务上，使用 Pegasos 优化线性 SVM。
- 在高维稀疏文本数据上，使用稀疏矩阵训练 RCV1 和 Amazon Review Polarity。
- 在非线性数据上，使用 SMO 训练 RBF kernel SVM，并与线性 SVM 做决策边界对比。
- 在多分类任务上，使用 One-vs-Rest 将二分类 SVM 扩展为多分类模型。

核心代码位置：

- `src/pegasos.py`：线性 Pegasos SVM 和 One-vs-Rest 多分类封装。
- `src/kernel_svm.py`：基于简化 SMO 的核 SVM，支持 RBF kernel。
- `src/datasets.py`：合成数据、two-moons 非线性数据、标准化和划分工具。
- `scripts/run_demo.py`：合成数据可视化实验。
- `scripts/run_real_datasets.py`：RCV1 和 Amazon Review Polarity 真实文本数据实验。
- `scripts/run_nonlinear_svm.py`：线性 SVM 与 RBF kernel SVM 的非线性分类对比实验。

## 2. SVM 二分类数学原理

给定训练集：

```math
\{(x_i, y_i)\}_{i=1}^{n}, \quad x_i \in \mathbb{R}^d,\quad y_i \in \{-1,+1\}
```

线性分类器为：

```math
f(x) = w^T x + b
```

预测规则为：

```math
\hat{y} = \operatorname{sign}(f(x))
```

SVM 的核心思想是最大化分类间隔。硬间隔 SVM 要求所有样本满足：

```math
y_i(w^T x_i + b) \ge 1
```

并最小化：

```math
\frac{1}{2}\|w\|^2
```

当数据不能完全线性分开时，引入松弛变量和惩罚系数，得到软间隔 SVM：

```math
\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
```

约束为：

```math
y_i(w^T x_i+b) \ge 1-\xi_i,\quad \xi_i \ge 0
```

也可以写成 hinge loss 形式：

```math
\min_{w,b} \frac{\lambda}{2}\|w\|^2 +
\frac{1}{n}\sum_{i=1}^{n}\max(0, 1-y_i(w^T x_i+b))
```

其中：

- `lambda_` 越大，正则越强，模型越简单。
- `C` 越大，对分类错误惩罚越强，模型更倾向拟合训练集。
- hinge loss 为 0 说明样本不仅分对了，而且在 margin 外侧。
- hinge loss 大于 0 说明样本违反了 margin，需要参与参数更新。

代码应用：

- `BinaryPegasosSVM.objective()` 计算上面的正则项和 hinge loss。
- `BinaryPegasosSVM.predict()` 使用 `sign(Xw+b)` 预测二分类标签。
- 训练日志中的 `active_rate` 表示违反 margin 的样本比例。

## 3. SMO 与核 SVM

### 3.1 对偶问题

SVM 的对偶形式为：

```math
\max_{\alpha}
\sum_{i=1}^{n}\alpha_i -
\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}
\alpha_i\alpha_j y_i y_j K(x_i,x_j)
```

约束为：

```math
0 \le \alpha_i \le C,\quad
\sum_{i=1}^{n}\alpha_i y_i = 0
```

分类函数为：

```math
f(x)=\sum_{i=1}^{n}\alpha_i y_i K(x_i,x)+b
```

只有 `alpha_i > 0` 的样本会参与预测，这些样本称为支持向量。

### 3.2 核函数

核函数把内积替换为隐式高维空间中的相似度：

```math
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
```

本项目新增的非线性分类使用 RBF kernel：

```math
K(x_i,x_j)=\exp(-\gamma \|x_i-x_j\|^2)
```

RBF kernel 能形成弯曲的决策边界，因此适合 two-moons、同心圆等非线性数据。

### 3.3 SMO 更新思想

SMO 每次选择两个拉格朗日乘子 `alpha_i` 和 `alpha_j` 更新，其余变量固定。这样可以在满足约束的情况下，把大规模二次规划拆成很多二维小问题。

简化 SMO 的主要步骤：

1. 计算样本误差：

```math
E_i=f(x_i)-y_i
```

2. 检查 KKT 条件，判断该样本是否需要更新。

3. 随机选择另一个样本 `j`。

4. 根据标签关系计算 `alpha_j` 的上下界 `L` 和 `H`。

5. 更新 `alpha_j`：

```math
\alpha_j \leftarrow \alpha_j -
\frac{y_j(E_i-E_j)}{2K_{ij}-K_{ii}-K_{jj}}
```

6. 将 `alpha_j` 裁剪到 `[L,H]`。

7. 根据等式约束更新 `alpha_i`：

```math
\alpha_i \leftarrow \alpha_i + y_i y_j(\alpha_j^{old}-\alpha_j)
```

8. 更新偏置 `b`。

代码应用：

- `src/kernel_svm.py` 中的 `BinaryKernelSVM.fit()` 实现简化 SMO。
- `BinaryKernelSVM._kernel_matrix()` 实现 linear、poly、rbf 三种核函数。
- `BinaryKernelSVM.support_vectors_` 保存支持向量。
- `scripts/run_nonlinear_svm.py` 使用 RBF kernel 在 two-moons 数据上训练非线性 SVM。

适用场景：

- SMO + RBF kernel 适合小到中等规模的非线性数据。
- 它需要计算核矩阵，内存复杂度约为 `O(n^2)`。
- 对 Amazon、RCV1 这种百万级高维文本数据，不适合直接使用核 SVM。

## 4. Pegasos 优化算法

Pegasos 直接优化 primal 形式：

```math
\min_w \frac{\lambda}{2}\|w\|^2 +
\frac{1}{n}\sum_{i=1}^{n}\max(0,1-y_i(w^Tx_i+b))
```

学习率为：

```math
\eta_t=\frac{1}{\lambda t}
```

单样本更新为：

```math
w \leftarrow (1-\eta_t\lambda)w
```

如果样本违反 margin：

```math
y_i(w^Tx_i+b)<1
```

则额外执行：

```math
w \leftarrow w+\eta_t y_i x_i
```

本项目使用 mini-batch 版本。对 batch 中违反 margin 的样本求平均修正：

```math
w \leftarrow (1-\eta_t\lambda)w+
\eta_t \frac{1}{|A|}\sum_{i \in A}y_i x_i
```

其中 `A` 是当前 batch 中违反 margin 的样本集合。

代码应用：

- `BinaryPegasosSVM.fit()` 中每个 batch 计算：

```python
margins = yb * (Xb @ self.w_ + self.b_)
active = margins < 1.0
```

- `active=True` 的样本参与 hinge loss 更新。
- 对稀疏矩阵，代码使用 `X.T @ y` 计算修正项，避免把文本特征转成 dense array。
- 训练日志输出：

```text
[Pegasos] epoch=1/5 seconds=... active_rate=... objective=... sample_acc=...
```

这些字段分别表示 epoch 耗时、违反 margin 比例、目标函数值、训练子样本准确率。

复杂度特点：

- 对 dense 数据，单轮复杂度约为 `O(nd)`。
- 对稀疏文本数据，复杂度接近 `O(nnz(X))`。
- 不需要核矩阵，适合 RCV1、Amazon Review Polarity 这类高维大规模文本分类任务。

## 5. 从二分类到多分类

SVM 原始形式是二分类模型。多分类可以通过 One-vs-Rest 扩展。

假设有 `K` 个类别：

```math
\mathcal{Y}=\{1,2,\dots,K\}
```

One-vs-Rest 会训练 `K` 个二分类器。第 `k` 个分类器把类别 `k` 当作正类，其余类别当作负类：

```math
y_i^{(k)} =
\begin{cases}
+1, & y_i=k \\
-1, & y_i \ne k
\end{cases}
```

预测时计算所有分类器的得分：

```math
s_k(x)=w_k^Tx+b_k
```

选择得分最高的类别：

```math
\hat{y}=\arg\max_k s_k(x)
```

代码应用：

- `OneVsRestPegasosSVM.fit()` 遍历所有类别，并为每个类别训练一个 `BinaryPegasosSVM`。
- `OneVsRestPegasosSVM.decision_function()` 返回所有类别的得分矩阵。
- `OneVsRestPegasosSVM.predict()` 使用 `argmax` 得到最终类别。
- `scripts/run_demo.py` 中的多分类混淆矩阵展示了该扩展方式的效果。

## 6. 实验流程

### 6.1 合成线性数据实验

运行命令：

```bash
python scripts/run_demo.py
```

输出图像：

- `output/figures/loss_curve.png`
- `output/figures/decision_boundary.png`
- `output/figures/sample_size_time.png`
- `output/figures/sample_size_accuracy.png`
- `output/figures/multiclass_confusion_matrix.png`

图像分析：

- `loss_curve.png`：目标函数随 epoch 下降，说明 Pegasos 在逐步优化 hinge loss 和正则项。
- `decision_boundary.png`：二维线性数据中，直线决策边界能较好分离两类样本，虚线表示 margin。
- `sample_size_time.png`：训练时间随样本规模增长，体现 Pegasos 近似线性扩展能力。
- `sample_size_accuracy.png`：样本数增加后准确率通常更稳定，但不是无限上升，受噪声和模型容量影响。
- `multiclass_confusion_matrix.png`：对角线越集中，说明 One-vs-Rest 多分类效果越好。

### 6.2 真实文本数据实验

运行 RCV1：

```bash
python scripts/run_real_datasets.py --dataset rcv1 --epochs 5 --batch-size 512
```

运行 Amazon Review Polarity 抽样实验：

```bash
python scripts/run_real_datasets.py --dataset amazon --epochs 5 --batch-size 512
```

运行 Amazon Review Polarity 全量实验：

```bash
python scripts/run_real_datasets.py --dataset amazon --amazon-full --epochs 5 --batch-size 2048 --hf-timeout 300
```

输出文件：

- `output/results/real_dataset_results.csv`
- `output/figures/real_dataset_report.png`
- `output/figures/real_dataset_training_curves.png`

图像分析：

- `real_dataset_report.png` 汇总真实数据集的测试准确率、训练时间、目标函数曲线和 margin 违反率。
- 测试准确率用于比较模型泛化效果。
- 训练时间体现稀疏矩阵和 Pegasos 对大规模文本数据的效率优势。
- 目标函数下降说明优化过程有效。
- `active_rate` 下降说明越来越多样本被正确分类并位于 margin 外侧。

### 6.3 非线性 SVM 实验

运行命令：

```bash
python scripts/run_nonlinear_svm.py
```

输出文件：

- `output/results/nonlinear_svm_results.csv`
- `output/figures/nonlinear_svm_decision_boundary.png`
- `output/figures/nonlinear_svm_training_diagnostics.png`

图像分析：

- `nonlinear_svm_decision_boundary.png` 对比线性 Pegasos SVM 和 RBF kernel SVM。
- two-moons 数据本身不是线性可分的，因此线性 SVM 只能画出直线边界，会出现系统性错误。
- RBF kernel SVM 可以形成弯曲边界，更贴合 two-moons 数据结构。
- 图中的空心圈表示支持向量，它们主要分布在类别边界附近，对决策边界起决定作用。
- `nonlinear_svm_training_diagnostics.png` 展示 SMO 迭代过程中的训练准确率和支持向量数量变化。

## 7. Amazon Review Polarity 为什么默认只训练 5 万个样本

Amazon Review Polarity 本身是百万级数据集，常用版本包含：

- 训练集约 `3,600,000` 条评论。
- 测试集约 `400,000` 条评论。

本项目脚本默认使用：

```text
--max-train-samples 50000
--max-test-samples 10000
```

原因不是数据集只有 5 万，而是为了让课程项目首次运行更稳定：

1. Hugging Face 下载大文件时容易受网络影响，抽样能更快跑通流程。
2. 初次验证算法时，5 万训练样本已经能体现高维稀疏文本分类的特点。
3. Amazon 全量文本向量化后会占用更多内存，尤其是保存稀疏矩阵和标签时。
4. `5` 个 epoch 是默认演示参数，不是最优训练参数；正式实验可以增加 epoch 或样本数。

如果要强调百万级规模，可以使用：

```bash
python scripts/run_real_datasets.py --dataset amazon --amazon-full --epochs 5 --batch-size 2048 --hf-timeout 300
```

如果机器内存有限，可以逐步扩大：

```bash
python scripts/run_real_datasets.py --dataset amazon --max-train-samples 200000 --max-test-samples 50000
python scripts/run_real_datasets.py --dataset amazon --max-train-samples 500000 --max-test-samples 100000
```

汇报时可以说明：默认 5 万是工程上的快速实验配置；Amazon 数据集本身是百万级，项目提供 `--amazon-full` 用于全量实验。

## 8. 方法对比总结

| 方法 | 主要优化对象 | 是否支持核函数 | 适合数据规模 | 本项目代码 |
| --- | --- | --- | --- | --- |
| 线性 SVM | primal 或 dual | 不使用核 | 中小规模到大规模 | `BinaryPegasosSVM` |
| SMO | dual 中的 alpha | 支持核函数 | 小到中等规模 | `BinaryKernelSVM` |
| Pegasos | primal 中的 w | 通常用于线性 SVM | 大规模稀疏数据 | `BinaryPegasosSVM` |
| One-vs-Rest | 多个二分类器 | 取决于底层分类器 | 多分类任务 | `OneVsRestPegasosSVM` |

结论：

- 核 SVM 通过 RBF kernel 能处理非线性分类，但核矩阵带来 `O(n^2)` 内存开销。
- Pegasos 线性 SVM 不使用核矩阵，训练速度快，适合高维稀疏文本。
- 对 RCV1 和 Amazon Review Polarity，线性 Pegasos 是更合适的主方法。
- 对 two-moons 等低维非线性数据，RBF kernel SVM 更能展示 SVM 的非线性分类能力。

## 9. 参考资料

- scikit-learn RCV1 数据集文档：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html
- Hugging Face Amazon Polarity 数据集：https://huggingface.co/datasets/fancyzhx/amazon_polarity
- 原始 Amazon Review Polarity 数据说明：https://course.fast.ai/datasets
