import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_split_dataset(npz_file_path):
    """
    加载已分割的数据集
    
    参数:
    npz_file_path: NPZ文件路径，包含分割的数据集
    
    返回:
    包含训练集、验证集和测试集的字典
    """
    try:
        # 加载NPZ文件，添加allow_pickle=True参数
        data = np.load(npz_file_path, allow_pickle=True)
        
        # 创建结果字典
        result = {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'] if data['X_val'].size > 0 else None,
            'y_val': data['y_val'] if data['y_val'].size > 0 else None,
            'X_test': data['X_test'] if data['X_test'].size > 0 else None,
            'y_test': data['y_test'] if data['y_test'].size > 0 else None,
            'feature_names': data['feature_names']
        }
        
        print(f"成功加载数据: {npz_file_path}")
        return result
    except Exception as e:
        print(f"错误: 加载文件 {npz_file_path} 时发生异常: {str(e)}")
        return None

def clean_data(X, column_means=None):
    """
    清洗数据集中的特殊值和缺失值
    
    参数:
    X: 输入特征矩阵
    column_means: 用于填充缺失值的列均值。如果为None，则计算X的列均值
    
    返回:
    清洗后的数据和列均值
    """
    # 转换为DataFrame以便处理
    df = pd.DataFrame(X)
    
    # 替换特殊值为NaN
    df = df.replace('?', np.nan)
    
    # 转换为数值类型
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 计算或使用列均值填充缺失值
    if column_means is None:
        column_means = df.mean()
    
    # 填充缺失值
    df = df.fillna(column_means)
    
    return df.values, column_means

def standardize_data(data_dict, output_dir=None):
    """
    对数据集进行标准化处理
    
    参数:
    data_dict: 包含数据集的字典（从load_split_dataset函数返回）
    output_dir: 保存标准化后数据的目录，如果为None则不保存
    
    返回:
    标准化后的数据集字典
    """
    result = {}
    result['feature_names'] = data_dict['feature_names']
    
    # 1. 清洗训练集数据
    X_train, column_means = clean_data(data_dict['X_train'])
    print(f"已清洗训练集数据，形状: {X_train.shape}")
    
    # 2. 使用相同的列均值清洗验证集和测试集
    X_val = None
    if data_dict['X_val'] is not None:
        X_val, _ = clean_data(data_dict['X_val'], column_means)
        print(f"已清洗验证集数据，形状: {X_val.shape}")
    
    X_test = None
    if data_dict['X_test'] is not None:
        X_test, _ = clean_data(data_dict['X_test'], column_means)
        print(f"已清洗测试集数据，形状: {X_test.shape}")
    
    # 3. 标准化处理
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    print("已完成训练集标准化")
    
    X_val_std = None
    if X_val is not None:
        X_val_std = scaler.transform(X_val)
        print("已完成验证集标准化")
    
    X_test_std = None
    if X_test is not None:
        X_test_std = scaler.transform(X_test)
        print("已完成测试集标准化")
    
    # 4. 保存结果
    result['X_train'] = X_train_std
    result['y_train'] = data_dict['y_train']
    result['X_val'] = X_val_std
    result['y_val'] = data_dict['y_val']
    result['X_test'] = X_test_std
    result['y_test'] = data_dict['y_test']
    result['scaler'] = scaler
    
    # 5. 如果需要，保存标准化后的数据
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 构建输出文件名
        output_file = os.path.join(output_dir, 'standardized_data.npz')
        
        # 保存标准化后的数据
        np.savez(
            output_file,
            X_train=X_train_std,
            y_train=data_dict['y_train'],
            X_val=X_val_std if X_val_std is not None else np.array([]),
            y_val=data_dict['y_val'] if data_dict['y_val'] is not None else np.array([]),
            X_test=X_test_std if X_test_std is not None else np.array([]),
            y_test=data_dict['y_test'] if data_dict['y_test'] is not None else np.array([]),
            feature_names=data_dict['feature_names']
        )
        print(f"已保存标准化数据到: {output_file}")
    
    return result

def verify_standardization(standardized_data):
    """
    验证标准化效果
    
    参数:
    standardized_data: 标准化后的数据集字典
    """
    # 检查训练集的均值和标准差
    train_mean = np.mean(standardized_data['X_train'], axis=0)
    train_std = np.std(standardized_data['X_train'], axis=0)
    
    print("\n标准化验证:")
    print(f"训练集均值前5个特征: {train_mean[:5]}")
    print(f"训练集标准差前5个特征: {train_std[:5]}")
    
    # 绘制训练集第一个特征的分布
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(standardized_data['X_train'][:, 0], bins=30, alpha=0.7)
    plt.title('第一个特征的分布')
    plt.xlabel('值')
    plt.ylabel('频率')
    
    # 绘制热图显示前10个特征的相关性
    if standardized_data['X_train'].shape[1] >= 10:
        plt.subplot(1, 2, 2)
        corr_matrix = np.corrcoef(standardized_data['X_train'][:, :10], rowvar=False)
        plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar()
        plt.title('前10个特征的相关性')
        plt.xticks(range(10), range(1, 11))
        plt.yticks(range(10), range(1, 11))
    
    plt.tight_layout()
    plt.show()

def process_dataset(input_npz_path, output_dir=None, verify=True):
    """
    处理单个数据集的完整流程
    
    参数:
    input_npz_path: 输入的NPZ文件路径
    output_dir: 输出目录，如果为None则不保存
    verify: 是否验证标准化效果
    
    返回:
    标准化后的数据集
    """
    # 1. 加载数据
    data = load_split_dataset(input_npz_path)
    if data is None:
        return None
    
    # 2. 打印原始数据信息
    print("\n原始数据集信息:")
    print(f"训练集形状: X={data['X_train'].shape}, y={data['y_train'].shape}")
    if data['X_val'] is not None:
        print(f"验证集形状: X={data['X_val'].shape}, y={data['y_val'].shape}")
    if data['X_test'] is not None:
        print(f"测试集形状: X={data['X_test'].shape}, y={data['y_test'].shape}")
    
    # 3. 标准化数据
    print("\n开始标准化数据...")
    standardized_data = standardize_data(data, output_dir)
    
    # 4. 验证标准化效果
    if verify:
        verify_standardization(standardized_data)
    
    return standardized_data

def process_all_datasets(input_dir, output_dir=None, verify=False):
    """
    处理目录中的所有数据集
    
    参数:
    input_dir: 包含NPZ文件的目录
    output_dir: 输出目录，如果为None则不保存
    verify: 是否验证标准化效果
    
    返回:
    处理后的数据集字典
    """
    import glob
    
    # 查找所有NPZ文件
    npz_files = glob.glob(os.path.join(input_dir, '**', '*train_val_test_split.npz'), recursive=True)
    print(f"找到 {len(npz_files)} 个数据集文件")
    
    results = {}
    
    for npz_file in npz_files:
        # 提取数据集名称
        dataset_name = os.path.basename(os.path.dirname(npz_file))
        print(f"\n处理数据集: {dataset_name}")
        
        # 创建对应的输出目录
        dataset_output_dir = None
        if output_dir:
            dataset_output_dir = os.path.join(output_dir, dataset_name)
        
        # 处理数据集
        result = process_dataset(npz_file, dataset_output_dir, verify=verify)
        
        if result:
            results[dataset_name] = result
    
    print(f"\n所有数据集处理完成，共 {len(results)} 个")
    return results

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='对拆分好的数据集进行标准化处理')
    parser.add_argument('--input', type=str, help='输入的NPZ文件路径或目录', default='NASADefectDataset/Split')
    parser.add_argument('--output', type=str, help='输出目录', default='NASADefectDataset/Standardized')
    parser.add_argument('--verify', action='store_true', help='是否验证标准化效果')
    parser.add_argument('--dataset', type=str, help='指定数据集名称，例如CM1', default=None)
    
    args = parser.parse_args()
    
    print(f"脚本运行于目录: {os.getcwd()}")
    print(f"参数: {args}")
    
    # 处理单个数据集
    if args.dataset:
        npz_path = os.path.join(args.input, args.dataset, 'train_val_test_split.npz')
        output_dir = os.path.join(args.output, args.dataset) if args.output else None
        
        if os.path.exists(npz_path):
            process_dataset(npz_path, output_dir, verify=args.verify)
        else:
            print(f"错误: 找不到数据集文件 {npz_path}")
    
    # 处理单个文件
    elif os.path.isfile(args.input):
        output_dir = args.output if args.output else None
        process_dataset(args.input, output_dir, verify=args.verify)
    
    # 处理目录下的所有数据集
    else:
        process_all_datasets(args.input, args.output, verify=args.verify)

if __name__ == "__main__":
    main() 