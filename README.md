# Pegasos 线性 SVM 项目

这个项目实现了一个不依赖 `sklearn.svm` 的 **Pegasos 线性 SVM**，用于课程项目中的“大规模样本下 SVM 优化算法验证”。

项目包含：

- `BinaryPegasosSVM`：二分类线性 SVM，使用 mini-batch Pegasos / 随机次梯度下降训练。
- `OneVsRestPegasosSVM`：一对其余多分类封装，可以处理多分类任务。
- `BinaryKernelSVM`：使用简化 SMO 训练的核 SVM，支持 RBF kernel，用于小规模非线性分类实验。
- 合成数据实验：loss 曲线、决策边界、样本规模-训练时间曲线、样本规模-准确率曲线、多分类混淆矩阵。
- 稀疏矩阵支持：可直接训练 RCV1、Amazon Review Polarity 这类高维文本特征。
- 真实数据集脚本：可运行 RCV1 和 Amazon Review Polarity，并把结果保存为 CSV。
- LIBSVM 格式真实数据集模板：可用于 covtype.binary、ijcnn1 等数据集。

## 一、项目结构

```text
pegasos_svm_project/
├── README.md
├── requirements.txt
├── src/
│   ├── pegasos.py
│   ├── kernel_svm.py
│   ├── datasets.py
│   └── metrics.py
├── scripts/
│   ├── run_demo.py
│   ├── run_nonlinear_svm.py
│   ├── run_real_datasets.py
│   └── run_real_dataset_template.py
├── tests/
│   └── test_pegasos.py
└── output/
    ├── figures/
    └── results/
```

## 二、数学目标

二分类线性 SVM 的优化目标为：

```math
\min_w \frac{\lambda}{2}\|w\|^2 + \frac{1}{n}\sum_{i=1}^n \max(0, 1-y_i(w^Tx_i+b))
```

其中：

- `lambda_` 控制正则化强度。
- `max(0, 1-y_i(w^Tx_i+b))` 是 hinge loss。
- 当 `y_i(w^Tx_i+b) >= 1` 时，样本已经被正确分类并且位于 margin 外侧。
- 当 `y_i(w^Tx_i+b) < 1` 时，样本分类不够理想，需要参与更新。

Pegasos 的单样本更新思想为：

```math
\eta_t = \frac{1}{\lambda t}
```

若样本违反 margin：

```math
w \leftarrow (1-\eta_t\lambda)w + \eta_t y_i x_i
```

否则只做正则收缩：

```math
w \leftarrow (1-\eta_t\lambda)w
```

本项目代码中使用 mini-batch 版本，以便训练更稳定。

## 三、运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行 demo 并生成图片：

```bash
python scripts/run_demo.py
```

运行非线性 SVM 对比实验：

```bash
python scripts/run_nonlinear_svm.py
```

输出图片在：

```text
output/figures/
```

运行测试：

```bash
pytest -q
```

## 四、已生成的实验图片

本压缩包中已经包含一次测试运行生成的图片：

- `loss_curve.png`：训练目标函数随 epoch 下降的曲线。
- `decision_boundary.png`：二维二分类数据上的分类边界和 margin。
- `sample_size_time.png`：训练样本数量与训练时间关系。
- `sample_size_accuracy.png`：训练样本数量与测试准确率关系。
- `multiclass_confusion_matrix.png`：一对其余多分类 SVM 的混淆矩阵。
- `nonlinear_svm_decision_boundary.png`：线性 Pegasos 与 RBF kernel SVM 在 two-moons 非线性数据上的决策边界对比。
- `nonlinear_svm_training_diagnostics.png`：非线性 SVM 训练过程中的准确率和支持向量数量变化。

## 五、如何接真实大规模数据集

本项目已经支持直接运行以下两个真实文本分类数据集：

- `RCV1`：使用 `sklearn.datasets.fetch_rcv1` 下载，默认把 `CCAT` 主题转成二分类任务。
- `Amazon Review Polarity`：使用 Hugging Face `datasets` 下载，并用 `HashingVectorizer` 转成稀疏文本特征。

首次运行需要联网下载数据。默认只抽取部分样本，方便课程项目先跑通；需要更大规模时可以调大 `--max-train-samples` 和 `--max-test-samples`。

配置网络环境：
```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
```

运行 RCV1：

```bash
python scripts/run_real_datasets.py --dataset rcv1 --epochs 5 --batch-size 512
```

运行 Amazon Review Polarity：

```bash
python scripts/run_real_datasets.py --dataset amazon --epochs 5 --batch-size 512
```

两个数据集都运行：

```bash
python scripts/run_real_datasets.py --dataset both --epochs 5 --batch-size 512
```

结果会保存到：

```text
output/results/real_dataset_results.csv
output/figures/real_dataset_report.png
output/figures/real_dataset_training_curves.png
```

常用参数：

- `--max-train-samples`：训练样本数，默认 `50000`。
- `--max-test-samples`：测试样本数，默认 `10000`。
- `--verbose`：训练日志详细程度，`0` 静默、`1` 输出每个 epoch 摘要、`2` 输出 batch 调试信息，默认 `1`。
- `--log-every`：`--verbose 2` 时每隔多少个 batch 输出一次调试信息，默认 `20`。
- `--no-record-history`：关闭目标函数/训练子样本准确率记录；默认开启，方便画训练曲线。
- `--no-figures`：只保存 CSV，不生成汇报图片。
- `--rcv1-label`：RCV1 的正类标签，默认 `CCAT`。
- `--hf-timeout`：Hugging Face 下载超时时间，默认 `120` 秒。
- `--amazon-full`：使用 Amazon Review Polarity 全量规模，约 `3600000` 个训练样本和 `400000` 个测试样本。
- `--amazon-train-file`、`--amazon-test-file`：使用本地 Amazon Polarity 文件，支持 parquet/csv/json/jsonl。
- `--hash-features`：Amazon 文本哈希特征维度，默认 `262144`。
- `--lambda`、`--epochs`、`--batch-size`：Pegasos 训练参数。

如果 Amazon Polarity 下载时出现 `read operation timed out`、`SSL: UNEXPECTED_EOF_WHILE_READING` 等网络错误，可以先单独跑 RCV1：

```bash
python scripts/run_real_datasets.py --dataset rcv1 --epochs 5 --batch-size 512
```

也可以设置更长超时后重试 Amazon：

```bash
python scripts/run_real_datasets.py --dataset amazon --hf-timeout 300 --epochs 5 --batch-size 512
```

如果已经手动下载 Amazon Polarity 的 train/test parquet 文件，可以离线运行：

```bash
python scripts/run_real_datasets.py --dataset amazon --amazon-train-file data/amazon_polarity/train.parquet --amazon-test-file data/amazon_polarity/test.parquet
```

如果需要体现 Amazon Polarity 的百万级规模，可以使用：

```bash
python scripts/run_real_datasets.py --dataset amazon --amazon-full --epochs 5 --batch-size 2048 --hf-timeout 300
```

全量 Amazon 会显著增加下载时间和内存占用；如果机器内存不足，可以先用 `--max-train-samples 200000`、`500000` 逐步放大。

其他可选数据集：

- 通用分类：covtype.binary、SUSY、HIGGS。
- 回归扩展：YearPredictionMSD，但需要另写 SVR 版本。

如果数据是 LIBSVM 格式，可使用模板：

```bash
python scripts/run_real_dataset_template.py --path data/rcv1_train.binary --epochs 5 --batch-size 256
```

注意：`run_real_dataset_template.py` 主要用于普通 LIBSVM 文件；RCV1 和 Amazon Review Polarity 推荐使用 `run_real_datasets.py`，它会保留稀疏矩阵，不会把高维文本特征转成 dense array。

## 六、课程展示建议

PPT 可以按以下主线讲：

1. SVM 最大间隔思想。
2. 软间隔与 hinge loss。
3. Pegasos 为什么适合大规模数据。
4. 二分类实现。
5. One-vs-Rest 多分类扩展。
6. 实验：loss、决策边界、样本规模与训练时间/准确率。
7. 总结：核 SVM 适合中小规模非线性数据，Pegasos 线性 SVM 更适合大规模高维数据。
