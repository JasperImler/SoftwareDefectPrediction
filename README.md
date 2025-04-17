# 软件缺陷预测模型

本项目基于NASA数据集，利用机器学习技术构建软件缺陷预测模型，旨在识别软件模块中的潜在缺陷。

## 背景

本项目是软件测试课程的期末作业。

## 数据集

本项目使用NASA缺陷数据集，包含各种软件度量指标和缺陷信息：
- 代码行数
- McCabe复杂度指标
- Halstead复杂度指标
- 代码变更指标
- 缺陷标签（有缺陷/无缺陷）

## Project Structure

```
├── arff_to_csv_converter.py    # Tool to convert ARFF files to CSV format
├── data_loader.py              # Utility for loading and preprocessing data
├── data_splitter.py            # Tool for splitting data into train/test sets
├── data_standardizer.py        # Data standardization tool
├── model_tuner.py              # Model parameter tuning with cross-validation
├── model_tuner_interactive.py  # Interactive interface for model tuning
├── OriginalData/               # Original NASA datasets in ARFF format
├── CleanedData/                # Cleaned datasets
├── CSV/                        # Datasets converted to CSV format
├── Split/                      # Train-validation-test split datasets
├── Standardized/               # Standardized datasets
├── Models/                     # Trained models and evaluation results
└── README.md
```

## 工作流程

项目完整工作流程包括以下步骤：

1. **数据转换**：
   ```bash
   python arff_to_csv_converter.py
   ```
   将原始ARFF文件转换为CSV格式。

2. **数据加载与清洗**：
   ```bash
   python data_loader.py
   ```
   加载数据集，处理缺失值和基本预处理。

3. **训练-测试集划分**：
   ```bash
   python data_splitter.py --dataset CM1
   ```
   将数据集划分为训练集、验证集和测试集。

4. **数据标准化**：
   ```bash
   python data_standardizer.py --dataset CM1 --verify
   ```
   标准化数据集并验证结果。

5. **模型调优**：
   ```bash
   python model_tuner.py --input Standardized/CM1/standardized_data.npz --model rf
   ```
   使用网格搜索和交叉验证调优随机森林模型。

6. **交互式模型调优**：
   ```bash
   python model_tuner_interactive.py
   ```
   启动交互式界面进行数据集选择和模型调优。

## 功能特点

1. **数据处理流程**：
   - ARFF转CSV格式
   - 数据加载、清洗与预处理
   - 训练-验证-测试集划分
   - 特征标准化及验证

2. **模型参数调优**：
   - 使用网格搜索和交叉验证进行超参数优化
   - 支持多种机器学习算法（随机森林、SVM、逻辑回归等）
   - 进度跟踪与可视化

3. **模型评估**：
   - 全面的性能指标（准确率、精确率、召回率、F1、AUC等）
   - 结果可视化（混淆矩阵、ROC曲线等）

4. **用户界面**：
   - 交互式命令行界面
   - 批处理能力
   - 可自定义评估标准

## 高级选项

- **快速模式**：
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model rf --reduce_params
  ```
  减少参数组合以加快训练速度。

- **随机搜索**：
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model svm --random --n_iter 20
  ```
  使用随机搜索代替网格搜索。

- **自定义评分标准**：
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model lr --scoring roc_auc
  ```
  使用AUC-ROC作为评分标准。

## 环境要求

- Python 3.6+
- NumPy, Pandas, Scikit-learn
- Matplotlib, tqdm, Seaborn(可选)

## 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib tqdm seaborn
```

## 结果输出

模型生成全面的评估报告，包括分类报告、混淆矩阵、ROC曲线等。结果保存在`Models/{dataset}/{model}/`目录下。

## 未来工作

- 实现集成学习方法
- 探索深度学习方法进行缺陷预测
- 特征重要性分析
- 软件演化的时间序列评估

## 许可

本项目仅供教育目的，使用MIT许可证。

## 致谢

- NASA提供的缺陷数据集
- Scikit-learn团队的机器学习库
- 所有项目贡献者
