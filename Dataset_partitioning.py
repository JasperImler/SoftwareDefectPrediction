import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob

def load_csv_data(csv_path):
    """
    加载CSV文件数据，提取特征值和目标值
    
    参数:
        csv_path: CSV文件路径
    
    返回:
        X: 特征值数据框
        y: 目标值数组
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 最后一列是目标值（Defective），其他列是特征值
    X = df.iloc[:, :-1]  # 所有行，除最后一列外的所有列
    y = df.iloc[:, -1]   # 所有行，最后一列
    
    # 将目标值转换为数值型（Y=1, N=0）
    if y.dtype == 'object':
        y = y.map({'buggy': 1, 'clean': 0})
    
    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    将数据按照指定比例划分为训练集、验证集和测试集
    
    参数:
        X: 特征值数据框
        y: 目标值数组
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.2
        test_ratio: 测试集比例，默认0.1
        random_state: 随机种子，默认42
    
    返回:
        x_train, x_val, x_test: 划分后的特征值数据集
        y_train, y_val, y_test: 划分后的目标值数据集
    """
    # 检查比例之和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 首先将数据分为训练集和临时集（验证集+测试集）
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=y
    )
    
    # 然后将临时集分为验证集和测试集
    # 计算验证集在临时集中的比例
    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
    
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(1 - val_ratio_in_temp), random_state=random_state, stratify=y_temp
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def process_csv_files(csv_dir):
    """
    处理指定目录下的所有CSV文件，提取特征值和目标值，并划分数据集
    
    参数:
        csv_dir: 包含CSV文件的目录路径
    
    返回:
        dataset_dict: 包含所有数据集的字典
    """
    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"在 {csv_dir} 中未找到CSV文件")
        return None
    
    # 创建一个字典来存储所有数据集
    dataset_dict = {}
    
    for csv_file in csv_files:
        # 获取文件名（不含扩展名）作为数据集名称
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"处理数据集: {dataset_name}")
        
        # 加载数据
        X, y = load_csv_data(csv_file)
        
        # 划分数据集
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)
        
        # 将划分后的数据集存储到字典中
        dataset_dict[dataset_name] = {
            'x_train': x_train, 'y_train': y_train,
            'x_val': x_val, 'y_val': y_val,
            'x_test': x_test, 'y_test': y_test
        }
        
        # 打印数据集大小信息
        print(f"  训练集: {x_train.shape[0]} 样本")
        print(f"  验证集: {x_val.shape[0]} 样本")
        print(f"  测试集: {x_test.shape[0]} 样本")
        print(f"  特征数: {x_train.shape[1]}")
        print(f"  正样本比例: 训练集 {y_train.mean():.2f}, 验证集 {y_val.mean():.2f}, 测试集 {y_test.mean():.2f}")
        print()
    
    return dataset_dict

def save_datasets(dataset_dict, output_dir):
    """
    保存划分后的数据集到指定目录
    
    参数:
        dataset_dict: 包含所有数据集的字典
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, datasets in dataset_dict.items():
        # 为每个数据集创建子目录
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 保存各个数据集
        for name, data in datasets.items():
            if name.startswith('x_'):
                # 保存特征值数据框
                data.to_csv(os.path.join(dataset_dir, f"{name}.csv"), index=False)
            else:  # y_train, y_val, y_test
                # 保存目标值数组
                pd.Series(data).to_csv(os.path.join(dataset_dir, f"{name}.csv"), index=False)
    
    print(f"所有数据集已保存到 {output_dir}")

def get_eigen_target(csv_dir, save_to_disk=False, output_dir=None):
    """
    主函数：处理CSV文件，提取特征值和目标值，划分数据集
    
    参数:
        csv_dir: 包含CSV文件的目录路径
        save_to_disk: 是否将数据集保存到磁盘，默认False
        output_dir: 输出目录路径，如果save_to_disk为True但未指定此参数，则使用默认路径
    
    返回:
        dataset_dict: 包含所有数据集的字典
    """
    # 处理CSV文件
    dataset_dict = process_csv_files(csv_dir)
    
    if dataset_dict is None:
        return None
    
    # 如果需要保存到磁盘
    if save_to_disk:
        if output_dir is None:
            # 默认输出目录
            output_dir = os.path.join(os.path.dirname(csv_dir), "SplitDatasets")
        
        save_datasets(dataset_dict, output_dir)
    
    return dataset_dict

if __name__ == "__main__":
    # CSV文件目录
    csv_dir = r"C:\Users\86159\Desktop\软件测试\期末大作业\SoftwareHomeWork\CSV_Data"
    
    # 处理数据并保存到磁盘
    get_eigen_target(csv_dir, save_to_disk=True)