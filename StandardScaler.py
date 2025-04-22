import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_dataset(dataset_dir, output_dir):
    """对指定数据集目录下的训练集和测试集进行标准化
    
    参数:
        dataset_dir: 数据集目录路径，包含x_train.csv, x_test.csv等文件
        output_dir: 标准化后数据的保存目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 读取训练集和测试集
    x_train = pd.read_csv(os.path.join(dataset_dir, 'x_train.csv'))
    x_test = pd.read_csv(os.path.join(dataset_dir, 'x_test.csv'))
    
    # 创建StandardScaler对象
    scaler = StandardScaler()
    
    # 使用训练集数据拟合scaler并转换
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns
    )
    
    # 使用相同的scaler转换测试集和验证集
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns
    )
 
    # 保存标准化后的数据到新目录
    x_train_scaled.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
    x_test_scaled.to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)

def standardize_all_datasets(base_dir, output_base_dir):
    """标准化所有数据集
    
    参数:
        base_dir: 包含所有数据集目录的基础目录
        output_base_dir: 标准化后数据集的保存基础目录
    """
    # 获取所有数据集目录
    dataset_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
    
    for dataset_name in dataset_dirs:
        print(f'正在标准化数据集: {dataset_name}')
        dataset_dir = os.path.join(base_dir, dataset_name)
        output_dir = os.path.join(output_base_dir, dataset_name)
        standardize_dataset(dataset_dir, output_dir)
        print(f'完成数据集 {dataset_name} 的标准化')

if __name__ == '__main__':
    # 设置包含所有数据集的基础目录
    base_dir = os.path.join(os.path.dirname(__file__), 'SplitDatasets')
    # 设置标准化后数据集的保存目录
    output_base_dir = os.path.join(os.path.dirname(__file__), 'StandardizedDatasets')
    standardize_all_datasets(base_dir, output_base_dir)