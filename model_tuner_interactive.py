import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印脚本标题"""
    print("\n" + "="*80)
    print("                    模型参数调优交互式执行工具")
    print("="*80)

def print_model_options():
    """打印可用的机器学习模型选项"""
    print("\n可用的机器学习模型:")
    print("1. 随机森林 (Random Forest)")
    print("2. 支持向量机 (SVM)")
    print("3. 逻辑回归 (Logistic Regression)")
    print("4. K近邻 (KNN)")
    print("5. 决策树 (Decision Tree)")
    print("6. 梯度提升 (Gradient Boosting)")
    
def print_tuning_options():
    """打印可用的调优选项"""
    print("\n调优方式:")
    print("1. 使用网格搜索调优随机森林模型")
    print("2. 使用随机搜索调优SVM模型")
    print("3. 使用AUC作为评分标准调优逻辑回归模型")
    print("4. 使用10折交叉验证调优KNN模型")
    print("5. 调优但不保存模型")
    print("6. 快速模式（减少参数组合数以加快训练）")
    print("7. 自定义调优参数")
    print("0. 退出程序")

def get_datasets():
    """获取标准化数据集列表"""
    base_dir = Path("NASADefectDataset/Standardized")
    
    if not base_dir.exists():
        print(f"错误: 目录 '{base_dir}' 不存在。请先运行数据标准化脚本。")
        return []
    
    datasets = []
    for dataset_dir in base_dir.iterdir():
        if dataset_dir.is_dir():
            npz_file = dataset_dir / "standardized_data.npz"
            if npz_file.exists():
                datasets.append((dataset_dir.name, str(npz_file)))
    
    return datasets

def print_datasets(datasets):
    """打印可用的数据集列表"""
    if not datasets:
        print("没有找到标准化数据集。请先运行数据标准化脚本。")
        return False
    
    print("\n可用的标准化数据集:")
    for i, (name, path) in enumerate(datasets, 1):
        print(f"{i}. {name} ({path})")
    
    return True

def get_user_choice(prompt, min_val, max_val):
    """获取用户的数字选择"""
    while True:
        try:
            choice = input(prompt)
            # 如果用户输入q或exit，则退出
            if choice.lower() in ['q', 'exit']:
                print("用户选择退出程序。")
                sys.exit(0)
            
            choice = int(choice)
            if min_val <= choice <= max_val:
                return choice
            else:
                print(f"请输入 {min_val} 到 {max_val} 之间的数字。")
        except ValueError:
            print("请输入有效的数字。")

def get_custom_parameters():
    """获取用户自定义的调优参数"""
    # 选择模型
    print_model_options()
    model_choice = get_user_choice("\n请选择要使用的模型 [1-6]: ", 1, 6)
    model_names = ['rf', 'svm', 'lr', 'knn', 'dt', 'gb']
    model = model_names[model_choice - 1]
    
    # 选择搜索方法
    search_method = input("\n使用随机搜索而非网格搜索? (y/n, 默认: n): ").lower()
    use_random_search = search_method in ['y', 'yes']
    
    # 如果使用随机搜索，设置迭代次数
    n_iter = 10
    if use_random_search:
        while True:
            try:
                n_iter = int(input("\n随机搜索迭代次数 (默认: 10): ") or "10")
                if n_iter > 0:
                    break
                else:
                    print("请输入大于0的数字。")
            except ValueError:
                print("请输入有效的数字。")
    
    # 设置交叉验证折数
    while True:
        try:
            cv = int(input("\n交叉验证折数 (默认: 5): ") or "5")
            if cv >= 2:
                break
            else:
                print("交叉验证折数必须至少为2。")
        except ValueError:
            print("请输入有效的数字。")
    
    # 选择评分标准
    scoring_options = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1_micro', 'f1_macro']
    print("\n可用的评分标准:")
    for i, scoring in enumerate(scoring_options, 1):
        print(f"{i}. {scoring}")
    
    scoring_choice = get_user_choice("\n请选择评分标准 [1-7]: ", 1, 7)
    scoring = scoring_options[scoring_choice - 1]
    
    # 是否减少参数组合数
    reduce_params = input("\n是否减少参数组合数以加快训练? (y/n, 默认: n): ").lower() in ['y', 'yes']
    
    # 是否保存模型
    save_model = input("\n是否保存模型? (y/n, 默认: y): ").lower()
    no_save = save_model in ['n', 'no']
    
    return model, use_random_search, n_iter, cv, scoring, reduce_params, no_save

def execute_tuning_option(option, dataset_path):
    """执行所选的调优选项"""
    cmd = [sys.executable, "model_tuner.py", "--input", dataset_path]
    
    if option == 1:  # 使用网格搜索调优随机森林模型
        cmd.extend(["--model", "rf"])
    elif option == 2:  # 使用随机搜索调优SVM模型
        cmd.extend(["--model", "svm", "--random", "--n_iter", "20"])
    elif option == 3:  # 使用AUC作为评分标准调优逻辑回归模型
        cmd.extend(["--model", "lr", "--scoring", "roc_auc"])
    elif option == 4:  # 使用10折交叉验证调优KNN模型
        cmd.extend(["--model", "knn", "--cv", "10"])
    elif option == 5:  # 调优但不保存模型
        model = input("请输入要调优的模型 (rf, svm, lr, knn, dt, gb): ").strip()
        if not model:
            model = "dt"  # 默认使用决策树
        cmd.extend(["--model", model, "--no_save"])
    elif option == 6:  # 快速模式（减少参数组合数）
        print("\n快速模式将减少参数组合数以加快训练")
        model = input("请输入要调优的模型 (rf, svm, lr, knn, dt, gb, 默认: dt): ").strip() or "dt"
        random_search = input("是否使用随机搜索? (y/n, 默认: y): ").lower() in ['', 'y', 'yes']
        
        cmd.extend(["--model", model, "--reduce_params"])
        if random_search:
            n_iter = input("随机搜索迭代次数 (默认: 10): ").strip() or "10"
            cmd.extend(["--random", "--n_iter", n_iter])
    elif option == 7:  # 自定义调优参数
        model, use_random_search, n_iter, cv, scoring, reduce_params, no_save = get_custom_parameters()
        
        cmd.extend(["--model", model])
        if use_random_search:
            cmd.extend(["--random", "--n_iter", str(n_iter)])
        if cv != 5:  # 默认值是5，如果不是5才需要指定
            cmd.extend(["--cv", str(cv)])
        if scoring != "f1":  # 默认值是f1，如果不是f1才需要指定
            cmd.extend(["--scoring", scoring])
        if reduce_params:
            cmd.append("--reduce_params")
        if no_save:
            cmd.append("--no_save")
    
    # 执行命令
    print("\n" + "="*80)
    print(f"执行命令: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    try:
        # 使用subprocess运行命令，将输出直接显示在终端
        process = subprocess.Popen(cmd)
        process.wait()
        
        if process.returncode == 0:
            print("\n命令执行成功完成！")
        else:
            print(f"\n命令执行失败，返回代码: {process.returncode}")
    except Exception as e:
        print(f"\n执行命令时发生错误: {str(e)}")

def main():
    """主函数"""
    clear_screen()
    print_header()
    
    # 获取并显示可用的数据集
    datasets = get_datasets()
    if not print_datasets(datasets):
        print("\n请先运行数据标准化脚本生成标准化数据集。")
        input("\n按Enter键退出...")
        return
    
    # 选择数据集
    dataset_choice = get_user_choice("\n请选择要使用的数据集 [1-{}]: ".format(len(datasets)), 1, len(datasets))
    selected_dataset = datasets[dataset_choice - 1]
    print(f"\n已选择数据集: {selected_dataset[0]}")
    
    while True:
        # 显示调优选项
        print_tuning_options()
        option = get_user_choice("\n请选择调优方式 [0-7]: ", 0, 7)
        
        if option == 0:
            print("\n感谢使用模型参数调优交互式执行工具！")
            break
        
        # 执行所选选项
        execute_tuning_option(option, selected_dataset[1])
        
        # 询问是否继续
        continue_choice = input("\n是否继续进行其他调优? (y/n, 默认: y): ").lower()
        if continue_choice in ['n', 'no']:
            print("\n感谢使用模型参数调优交互式执行工具！")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行。感谢使用！")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按Enter键退出...") 