# 软件缺陷预测项目

## 项目简介
本项目是一个基于机器学习的软件缺陷预测系统，用于预测Apache、Safe和Zxing等开源项目的代码缺陷。项目包含数据预处理、模型训练和评估三个主要模块。

## 原始数据集
源自https://github.com/ai-se/HDP_pyjnius/tree/master/dataset/Relink

### 文件格式转换
由于开源数据集文件为 arff 格式，而在使用 python pandas 读取数
据时一般使用 read_csv 方法读取 csv 格式的文件，所以需要将其转化为可以使用的 csv 格
式。
输入：软件缺陷数据集 NASADefectDataset 的 arff 文件 
输出：软件缺陷数据集 NASADefectDataset 的 csv 文件
运行`aff_to_csv.py`进行格式转换

### 数据预处理
1. 运行`StandardScaler.py`进行数据标准化
2. 运行`Dataset_partitioning.py`划分训练集和测试集

### 模型训练与评估
2. 运行`GridSearchCV.py`进行超参数优化
3. 运行`train.py`训练模型并生成评估报告

## 项目结构
```
├── CSV_Data/                # 原始csv数据集
├── SplitDatasets/           # 划分后的数据集
├── StandardizedDatasets/    # 标准化后的数据集
├── OptimizationResults/     # 超参数优化结果
├── EvaluationResults/       # 模型评估报告
├── getModel.py              # 模型初始化
├── train.py                 # 模型训练与评估
└── README.md                # 项目说明文档
```
