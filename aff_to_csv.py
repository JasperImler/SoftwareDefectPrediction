import os
import csv
import re
import glob

def arff_to_csv(arff_path, csv_path):
    """
    将 ARFF 文件转换为 CSV 文件
    
    参数:
        arff_path: ARFF 文件路径
        csv_path: 输出的 CSV 文件路径
    """
    # 读取 ARFF 文件内容
    with open(arff_path, 'r', encoding='utf-8') as arff_file:
        content = arff_file.read()
    
    # 提取数据部分
    data_match = re.search(r'@data\s*([\s\S]*)', content, re.IGNORECASE)
    if not data_match:
        print(f"错误: 在 {arff_path} 中未找到 @data 部分")
        return False
    
    data_section = data_match.group(1).strip()
    
    # 提取属性名称
    attribute_pattern = r'@attribute\s+([^\s{]+)'
    attributes = re.findall(attribute_pattern, content, re.IGNORECASE)
    
    if not attributes:
        print(f"错误: 在 {arff_path} 中未找到属性定义")
        return False
    
    # 写入 CSV 文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 写入表头
        csv_writer.writerow(attributes)
        
        # 写入数据行
        for line in data_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('%'):  # 跳过空行和注释行
                # 处理引号内的逗号
                processed_line = []
                in_quotes = False
                current_field = ""
                
                for char in line:
                    if char == '"' or char == "'":
                        in_quotes = not in_quotes
                        current_field += char
                    elif char == ',' and not in_quotes:
                        processed_line.append(current_field.strip())
                        current_field = ""
                    else:
                        current_field += char
                
                if current_field:  # 添加最后一个字段
                    processed_line.append(current_field.strip())
                
                csv_writer.writerow(processed_line)
    
    print(f"已将 {arff_path} 转换为 {csv_path}")
    return True

def convert_directory(input_dir, output_dir=None):
    """
    转换目录中的所有 ARFF 文件为 CSV 文件
    
    参数:
        input_dir: 包含 ARFF 文件的目录
        output_dir: 输出 CSV 文件的目录，如果为 None，则使用输入目录
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 ARFF 文件
    arff_files = glob.glob(os.path.join(input_dir, "*.arff"))
    arff_files.extend(glob.glob(os.path.join(input_dir, "*.ARFF")))
    
    if not arff_files:
        print(f"在 {input_dir} 中未找到 ARFF 文件")
        return
    
    success_count = 0
    for arff_file in arff_files:
        base_name = os.path.basename(arff_file)
        name_without_ext = os.path.splitext(base_name)[0]
        csv_file = os.path.join(output_dir, f"{name_without_ext}.csv")
        
        if arff_to_csv(arff_file, csv_file):
            success_count += 1
    
    print(f"转换完成: {success_count}/{len(arff_files)} 个文件已成功转换")

if __name__ == "__main__":
    # 指定包含 ARFF 文件的目录 D'
    input_directory = r"C:\Users\86159\Desktop\软件测试\期末大作业\SoftwareHomeWork\HDP_pyjnius\dataset\Relink"
    
    # 可以指定输出目录，如果不指定则使用相同目录
    output_directory = r"c:\Users\86159\Desktop\软件测试\期末大作业\SoftWareHomeWork\CSV_Data"
    
    convert_directory(input_directory, output_directory)