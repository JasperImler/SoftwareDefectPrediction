import os
import re
import csv
import glob

def arff_to_csv(arff_file_path, csv_file_path):
    """
    将ARFF文件转换为CSV文件
    
    参数:
    arff_file_path: ARFF文件的路径
    csv_file_path: 输出的CSV文件路径
    """
    # 读取ARFF文件
    with open(arff_file_path, 'r', encoding='utf-8') as arff_file:
        content = arff_file.read()
    
    # 解析ARFF文件
    # 查找@attribute行来获取属性名称
    attribute_pattern = re.compile(r'@attribute\s+(\S+)\s+.*', re.IGNORECASE)
    attributes = attribute_pattern.findall(content)
    
    # 查找@data之后的数据部分
    data_match = re.search(r'@data(.*)', content, re.DOTALL | re.IGNORECASE)
    if data_match:
        data_section = data_match.group(1).strip()
        # 分割数据行
        data_lines = [line.strip() for line in data_section.split('\n') if line.strip()]
        
        # 写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # 写入表头
            csv_writer.writerow(attributes)
            
            # 写入数据行
            for line in data_lines:
                # 解析并写入数据
                values = line.split(',')
                csv_writer.writerow(values)
        
        print(f"转换完成: {arff_file_path} -> {csv_file_path}")
        return True
    else:
        print(f"无法在文件中找到@data部分: {arff_file_path}")
        return False

def convert_directory(input_dir, output_dir, is_recursive=True):
    """
    批量转换目录中的所有ARFF文件为CSV文件
    
    参数:
    input_dir: 包含ARFF文件的输入目录
    output_dir: 保存CSV文件的输出目录
    is_recursive: 是否递归处理子目录
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确定搜索模式
    search_pattern = os.path.join(input_dir, '**', '*.arff') if is_recursive else os.path.join(input_dir, '*.arff')
    
    # 查找所有ARFF文件
    arff_files = glob.glob(search_pattern, recursive=is_recursive)
    
    success_count = 0
    failure_count = 0
    
    for arff_file in arff_files:
        # 创建相对路径的目录结构
        rel_path = os.path.relpath(arff_file, input_dir)
        output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.csv')
        
        # 确保输出文件的目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换文件
        if arff_to_csv(arff_file, output_path):
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\n转换完成! 成功: {success_count}, 失败: {failure_count}")

def main():
    """
    主函数，处理命令行参数并执行转换
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='将ARFF文件转换为CSV文件')
    parser.add_argument('--cleaned_data', type=str, help='CleanedData目录路径', default='NASADefectDataset/CleanedData')
    parser.add_argument('--original_data', type=str, help='OriginalData目录路径', default='NASADefectDataset/OriginalData')
    parser.add_argument('--output_dir', type=str, help='输出CSV文件的目录', default='NASADefectDataset/CSV')
    parser.add_argument('--no_recursive', action='store_true', help='不递归处理子目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    cleaned_output = os.path.join(args.output_dir, 'CleanedData')
    original_output = os.path.join(args.output_dir, 'OriginalData')
    
    # 转换CleanedData
    print(f"\n处理CleanedData目录: {args.cleaned_data}")
    convert_directory(args.cleaned_data, cleaned_output, not args.no_recursive)
    
    # 转换OriginalData
    print(f"\n处理OriginalData目录: {args.original_data}")
    convert_directory(args.original_data, original_output, not args.no_recursive)
    
    print("\n所有转换任务完成!")

if __name__ == "__main__":
    main() 