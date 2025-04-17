import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体支持
import matplotlib as mpl
import platform

# 根据操作系统选择合适的中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
elif system == 'Darwin':
    # macOS系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    # Linux系统
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 确保matplotlib能显示中文
mpl.rcParams['font.family'] = plt.rcParams['font.sans-serif'][0]

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
        
        return result
    except Exception as e:
        print(f"错误: 加载文件 {npz_file_path} 时发生异常: {str(e)}")
        return None

def preprocess_data(data_dict, normalize=True, apply_pca=False, n_components=0.95):
    """
    预处理数据集，包括标准化和可选的PCA降维
    
    参数:
    data_dict: 包含数据集的字典
    normalize: 是否标准化数据，默认为True
    apply_pca: 是否应用PCA降维，默认为False
    n_components: PCA组件数或保留方差比例，默认保留95%的方差
    
    返回:
    预处理后的数据字典
    """
    result = data_dict.copy()
    
    # 标准化特征前进行数据清洗
    if normalize:
        # 处理X_train中的特殊值
        X_train_df = pd.DataFrame(result['X_train'])
        X_train_df = X_train_df.replace('?', np.nan)
        X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce')
        
        # 计算每列的均值，用于填充缺失值
        column_means = X_train_df.mean()
        X_train_df = X_train_df.fillna(column_means)
        
        # 更新训练集
        result['X_train'] = X_train_df.values
        
        # 同样处理验证集和测试集
        if result['X_val'] is not None:
            X_val_df = pd.DataFrame(result['X_val'])
            X_val_df = X_val_df.replace('?', np.nan)
            X_val_df = X_val_df.apply(pd.to_numeric, errors='coerce')
            X_val_df = X_val_df.fillna(column_means)  # 使用训练集的均值
            result['X_val'] = X_val_df.values
        
        if result['X_test'] is not None:
            X_test_df = pd.DataFrame(result['X_test'])
            X_test_df = X_test_df.replace('?', np.nan)
            X_test_df = X_test_df.apply(pd.to_numeric, errors='coerce')
            X_test_df = X_test_df.fillna(column_means)  # 使用训练集的均值
            result['X_test'] = X_test_df.values
        
        # 原有的标准化代码
        scaler = StandardScaler()
        result['X_train'] = scaler.fit_transform(result['X_train'])
        
        if result['X_val'] is not None:
            result['X_val'] = scaler.transform(result['X_val'])
        
        if result['X_test'] is not None:
            result['X_test'] = scaler.transform(result['X_test'])
        
        # 保存scaler供后续使用
        result['scaler'] = scaler
    
    # 应用PCA降维
    if apply_pca:
        pca = PCA(n_components=n_components)
        result['X_train'] = pca.fit_transform(result['X_train'])
        
        if result['X_val'] is not None:
            result['X_val'] = pca.transform(result['X_val'])
        
        if result['X_test'] is not None:
            result['X_test'] = pca.transform(result['X_test'])
        
        # 保存PCA模型和解释方差比
        result['pca'] = pca
        result['explained_variance_ratio'] = pca.explained_variance_ratio_
        
        # 更新特征名称
        result['feature_names'] = [f'PC{i+1}' for i in range(pca.n_components_)]
    
    return result

def visualize_data(data_dict, save_path=None):
    """
    可视化数据集
    
    参数:
    data_dict: 包含数据集的字典
    save_path: 保存图像的路径，如果为None则不保存
    """
    # 如果数据已经进行了PCA降维，则绘制前两个主成分的散点图
    if 'pca' in data_dict:
        plt.figure(figsize=(10, 8))
        
        # 绘制训练集
        plt.scatter(
            data_dict['X_train'][:, 0], 
            data_dict['X_train'][:, 1], 
            c=data_dict['y_train'],
            cmap='viridis', 
            alpha=0.5, 
            s=50,
            label='训练集'
        )
        
        # 绘制验证集（如果存在）
        if data_dict['X_val'] is not None:
            plt.scatter(
                data_dict['X_val'][:, 0], 
                data_dict['X_val'][:, 1], 
                c=data_dict['y_val'],
                cmap='viridis', 
                marker='x', 
                s=70,
                label='验证集'
            )
        
        # 绘制测试集（如果存在）
        if data_dict['X_test'] is not None:
            plt.scatter(
                data_dict['X_test'][:, 0], 
                data_dict['X_test'][:, 1], 
                c=data_dict['y_test'],
                cmap='viridis', 
                marker='s', 
                s=70,
                label='测试集'
            )
        
        # 添加标题和标签
        plt.title('PCA降维后的数据可视化')
        plt.xlabel(f'主成分1 (解释方差: {data_dict["explained_variance_ratio"][0]:.2f})')
        plt.ylabel(f'主成分2 (解释方差: {data_dict["explained_variance_ratio"][1]:.2f})')
        plt.colorbar(label='目标变量')
        plt.legend()
        
        # 保存图像（如果需要）
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    # 绘制目标变量分布
    plt.figure(figsize=(10, 6))
    
    # 训练集
    train_counts = np.bincount(data_dict['y_train'].astype(int))
    train_labels = [f'类别{i}' for i in range(len(train_counts))]
    
    plt.subplot(1, 3, 1)
    plt.bar(train_labels, train_counts)
    plt.title('训练集目标变量分布')
    plt.ylabel('样本数')
    
    # 验证集（如果存在）
    if data_dict['y_val'] is not None:
        val_counts = np.bincount(data_dict['y_val'].astype(int))
        val_labels = [f'类别{i}' for i in range(len(val_counts))]
        
        plt.subplot(1, 3, 2)
        plt.bar(val_labels, val_counts)
        plt.title('验证集目标变量分布')
    
    # 测试集（如果存在）
    if data_dict['y_test'] is not None:
        test_counts = np.bincount(data_dict['y_test'].astype(int))
        test_labels = [f'类别{i}' for i in range(len(test_counts))]
        
        plt.subplot(1, 3, 3)
        plt.bar(test_labels, test_counts)
        plt.title('测试集目标变量分布')
    
    plt.tight_layout()
    
    # 保存图像（如果需要）
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        plt.savefig(f"{base_path}_distribution.png")
    
    plt.show()

def example_usage():
    """
    示例：如何使用数据加载和预处理函数
    """
    # 示例数据集路径
    data_path = 'NASADefectDataset/Split/CM1/train_val_test_split.npz'
    
    # 如果文件存在，加载并处理数据
    if os.path.exists(data_path):
        # 加载数据
        data = load_split_dataset(data_path)
        
        if data:
            print("原始数据集信息:")
            print(f"训练集形状: X={data['X_train'].shape}, y={data['y_train'].shape}")
            if data['X_val'] is not None:
                print(f"验证集形状: X={data['X_val'].shape}, y={data['y_val'].shape}")
            if data['X_test'] is not None:
                print(f"测试集形状: X={data['X_test'].shape}, y={data['y_test'].shape}")
            print(f"特征数量: {len(data['feature_names'])}")
            print(f"特征名称: {data['feature_names'][:5]}...")
            
            # 标准化数据
            norm_data = preprocess_data(data, normalize=True, apply_pca=False)
            print("\n标准化后数据:")
            print(f"训练集均值: {np.mean(norm_data['X_train'], axis=0)[:3]}...")
            print(f"训练集标准差: {np.std(norm_data['X_train'], axis=0)[:3]}...")
            
            # 标准化并应用PCA
            pca_data = preprocess_data(data, normalize=True, apply_pca=True, n_components=0.95)
            pca_components = pca_data['X_train'].shape[1] if 'pca' in pca_data else 0
            
            print("\nPCA降维后数据:")
            print(f"保留主成分数量: {pca_components}")
            print(f"训练集形状: {pca_data['X_train'].shape}")
            
            if 'explained_variance_ratio' in pca_data:
                cumulative_variance = np.cumsum(pca_data['explained_variance_ratio'])
                print(f"解释方差比例: {pca_data['explained_variance_ratio'][:5]}...")
                print(f"累积解释方差: {cumulative_variance[:5]}...")
            
            # 可视化数据
            try:
                visualize_data(pca_data, save_path='NASADefectDataset/Split/CM1/pca_visualization.png')
            except Exception as e:
                print(f"可视化数据时发生错误: {str(e)}")
        else:
            print(f"无法加载数据集: {data_path}")
    else:
        print(f"数据文件不存在: {data_path}")
        print("请先运行data_splitter.py生成分割后的数据集")

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='加载和预处理分割后的数据集')
    parser.add_argument('--data_path', type=str, help='NPZ数据文件路径')
    parser.add_argument('--normalize', action='store_true', help='是否标准化数据')
    parser.add_argument('--apply_pca', action='store_true', help='是否应用PCA降维')
    parser.add_argument('--n_components', type=float, default=0.95, help='PCA组件数或保留方差比例')
    parser.add_argument('--visualize', action='store_true', help='是否可视化数据')
    parser.add_argument('--save_path', type=str, help='保存可视化结果的路径')
    
    args = parser.parse_args()
    
    if args.data_path:
        # 加载数据
        data = load_split_dataset(args.data_path)
        
        if data:
            # 预处理数据
            processed_data = preprocess_data(
                data, 
                normalize=args.normalize,
                apply_pca=args.apply_pca,
                n_components=args.n_components
            )
            
            # 打印数据信息
            print("数据集信息:")
            print(f"训练集形状: X={processed_data['X_train'].shape}, y={processed_data['y_train'].shape}")
            if processed_data['X_val'] is not None:
                print(f"验证集形状: X={processed_data['X_val'].shape}, y={processed_data['y_val'].shape}")
            if processed_data['X_test'] is not None:
                print(f"测试集形状: X={processed_data['X_test'].shape}, y={processed_data['y_test'].shape}")
            
            # 可视化数据（如果需要）
            if args.visualize:
                visualize_data(processed_data, save_path=args.save_path)
        else:
            print(f"无法加载数据集: {args.data_path}")
    else:
        # 没有提供数据路径，运行示例
        example_usage()

if __name__ == "__main__":
    main() 