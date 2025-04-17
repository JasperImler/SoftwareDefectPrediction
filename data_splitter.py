import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(csv_file_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, target_column='Defective', random_state=42):
    """
    读取CSV文件并将数据集按照指定比例分割成训练集、验证集和测试集
    
    参数:
    csv_file_path: CSV文件路径
    train_ratio: 训练集比例，默认为0.7
    val_ratio: 验证集比例，默认为0.2
    test_ratio: 测试集比例，默认为0.1
    target_column: 目标变量列名，默认为'Defective'
    random_state: 随机种子，确保结果可复现，默认为42
    
    返回:
    包含训练集、验证集和测试集的特征和目标变量的字典
    """
    # 验证比例之和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须等于1"
    
    try:
        # 读取CSV文件
        print(f"读取文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"数据集形状: {df.shape}")
        
        # 检查目标列是否存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 在数据集中不存在")
        
        # 处理目标变量，转换为数值型
        if df[target_column].dtype == 'object':
            # 如果是分类变量，转换为数值
            print(f"将目标变量 '{target_column}' 转换为数值型")
            # 检查是否为二分类问题（如'Y'/'N'格式）
            if set(df[target_column].unique()) == {'Y', 'N'}:
                df[target_column] = df[target_column].map({'Y': 1, 'N': 0})
            else:
                # 对于多分类问题，可以使用LabelEncoder
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                df[target_column] = encoder.fit_transform(df[target_column])
                print(f"类别映射: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        # 分离特征和目标变量
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 先分离出测试集
        if test_ratio > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=random_state
            )
        else:
            X_temp, y_temp = X, y
            X_test, y_test = None, None
        
        # 再从剩余数据中分离出验证集和训练集
        if val_ratio > 0:
            # 计算验证集在剩余数据中的比例
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        # 打印各集合的大小
        print(f"训练集大小: {X_train.shape[0]} 样本")
        if X_val is not None:
            print(f"验证集大小: {X_val.shape[0]} 样本")
        if X_test is not None:
            print(f"测试集大小: {X_test.shape[0]} 样本")
        
        # 返回分割后的数据集
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': list(X.columns)
        }
    
    except Exception as e:
        print(f"错误: 处理文件 {csv_file_path} 时发生异常: {str(e)}")
        return None

def process_all_csv_files(csv_dir, output_dir=None, target_column='Defective'):
    """
    处理目录中的所有CSV文件，将其分割并可选地保存结果
    
    参数:
    csv_dir: 包含CSV文件的目录
    output_dir: 保存分割结果的目录，如果为None则不保存
    target_column: 目标变量列名
    """
    # 检查目录是否存在
    if not os.path.exists(csv_dir):
        print(f"错误: 目录不存在 {csv_dir}")
        return
    
    # 创建输出目录（如果需要）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, '**', '*.csv'), recursive=True)
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    results = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n处理文件: {filename}")
        
        # 分割数据集
        split_data = split_dataset(csv_file, target_column=target_column)
        
        if split_data:
            results[filename] = split_data
            
            # 保存结果（如果需要）
            if output_dir:
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, base_name)
                
                # 创建数据集目录
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                # 保存数据集
                np.savez(
                    os.path.join(output_path, 'train_val_test_split.npz'),
                    X_train=split_data['X_train'].values, 
                    y_train=split_data['y_train'].values,
                    X_val=split_data['X_val'].values if split_data['X_val'] is not None else np.array([]),
                    y_val=split_data['y_val'].values if split_data['y_val'] is not None else np.array([]),
                    X_test=split_data['X_test'].values if split_data['X_test'] is not None else np.array([]),
                    y_test=split_data['y_test'].values if split_data['y_test'] is not None else np.array([]),
                    feature_names=np.array(split_data['feature_names'])
                )
                print(f"已保存分割结果到: {output_path}")
    
    print(f"\n处理完成! 成功处理 {len(results)} 个文件")
    return results

def main():
    """
    主函数，处理命令行参数并执行数据分割
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='将CSV数据分割为训练集、验证集和测试集')
    parser.add_argument('--csv_dir', type=str, help='CSV文件目录', default='NASADefectDataset/CSV')
    parser.add_argument('--output_dir', type=str, help='输出分割结果的目录', default='NASADefectDataset/Split')
    parser.add_argument('--target_column', type=str, help='目标变量列名', default='Defective')
    parser.add_argument('--train_ratio', type=float, help='训练集比例', default=0.7)
    parser.add_argument('--val_ratio', type=float, help='验证集比例', default=0.2)
    parser.add_argument('--test_ratio', type=float, help='测试集比例', default=0.1)
    
    args = parser.parse_args()
    
    print(f"脚本运行于目录: {os.getcwd()}")
    print(f"参数: {args}")
    
    # 检查比例之和是否为1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-10:
        print("错误: 比例之和必须等于1")
        return
    
    # 处理单个文件的示例
    if os.path.isfile(args.csv_dir):
        split_data = split_dataset(
            args.csv_dir, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            target_column=args.target_column
        )
        
        if split_data and args.output_dir:
            base_name = os.path.splitext(os.path.basename(args.csv_dir))[0]
            output_path = os.path.join(args.output_dir, base_name)
            
            # 创建数据集目录
            os.makedirs(output_path, exist_ok=True)
            
            # 保存数据集
            np.savez(
                os.path.join(output_path, 'train_val_test_split.npz'),
                X_train=split_data['X_train'].values, 
                y_train=split_data['y_train'].values,
                X_val=split_data['X_val'].values if split_data['X_val'] is not None else np.array([]),
                y_val=split_data['y_val'].values if split_data['y_val'] is not None else np.array([]),
                X_test=split_data['X_test'].values if split_data['X_test'] is not None else np.array([]),
                y_test=split_data['y_test'].values if split_data['y_test'] is not None else np.array([]),
                feature_names=np.array(split_data['feature_names'])
            )
            print(f"已保存分割结果到: {output_path}")
    else:
        # 处理目录下的所有CSV文件
        process_all_csv_files(
            args.csv_dir,
            args.output_dir,
            target_column=args.target_column
        )

if __name__ == "__main__":
    main() 