import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import joblib
import sys
import traceback
import matplotlib as mpl
from tqdm import tqdm
import multiprocessing

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置字体大小

# 检查是否有中文字体可用
def check_chinese_font():
    """检查系统中可用的中文字体并设置"""
    print("检查中文字体支持...")
    fonts = mpl.font_manager.findSystemFonts()
    
    # 中文字体优先级列表
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
    
    # 检查哪些中文字体可用
    available_chinese_fonts = []
    for font in fonts:
        font_name = mpl.font_manager.FontProperties(fname=font).get_name()
        if any(chinese_font.lower() in font_name.lower() for chinese_font in chinese_fonts):
            available_chinese_fonts.append(font_name)
    
    if available_chinese_fonts:
        print(f"找到可用的中文字体: {available_chinese_fonts}")
        plt.rcParams['font.sans-serif'] = available_chinese_fonts + plt.rcParams['font.sans-serif']
        return True
    else:
        print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
        # 尝试使用不需要特定字体的Agg后端
        mpl.use('Agg')
        return False

# 在加载模块时检查中文字体
check_chinese_font()

def load_standardized_data(npz_file_path):
    """
    加载标准化的数据集
    
    参数:
    npz_file_path: NPZ文件路径，包含标准化的数据集
    
    返回:
    包含训练集、验证集和测试集的字典
    """
    print(f"尝试加载文件: {npz_file_path}")
    print(f"文件是否存在: {os.path.exists(npz_file_path)}")
    
    try:
        # 加载NPZ文件，添加allow_pickle=True参数
        data = np.load(npz_file_path, allow_pickle=True)
        
        # 打印文件中的数组名称
        print(f"文件中包含的数组: {list(data.keys())}")
        
        # 创建结果字典
        result = {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'] if data['X_val'].size > 0 else None,
            'y_val': data['y_val'] if data['y_val'].size > 0 else None,
            'X_test': data['X_test'] if data['X_test'].size > 0 else None,
            'y_test': data['y_test'] if data['y_test'].size > 0 else None,
            'feature_names': data['feature_names'] if 'feature_names' in data else None
        }
        
        # 打印加载的数据形状
        print(f"X_train 形状: {result['X_train'].shape}")
        print(f"y_train 形状: {result['y_train'].shape}")
        if result['X_val'] is not None:
            print(f"X_val 形状: {result['X_val'].shape}")
        if result['X_test'] is not None:
            print(f"X_test 形状: {result['X_test'].shape}")
        
        print(f"成功加载标准化数据: {npz_file_path}")
        return result
    except Exception as e:
        print(f"错误: 加载文件 {npz_file_path} 时发生异常: {str(e)}")
        print("异常追踪信息:")
        traceback.print_exc()
        return None

def get_model_and_param_grid(model_name, reduce_params=False):
    """
    根据模型名称返回模型实例和对应的参数网格
    
    参数:
    model_name: 模型名称，如'rf'代表随机森林
    reduce_params: 是否减少参数组合数以加快训练
    
    返回:
    model: 初始化的模型实例
    param_grid: 参数网格字典
    """
    if model_name == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    elif model_name == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
    elif model_name == 'dt':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_name == 'gb':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None]
        }
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    # 如果需要减少参数组合数
    if reduce_params:
        param_grid = reduce_param_grid_size(param_grid)
    
    # 计算参数组合总数
    combinations = 1
    for values in param_grid.values():
        combinations *= len(values)
    
    print(f"模型 {model_name} 的参数网格包含 {combinations} 种组合")
    
    return model, param_grid

def tune_model_parameters(model, param_grid, X_train, y_train, cv=5, scoring='f1', n_jobs=-1, verbose=1, random=False, n_iter=10):
    """
    使用交叉验证和网格搜索/随机搜索调整模型参数
    
    参数:
    model: 初始化的模型实例
    param_grid: 参数网格字典
    X_train: 训练集特征
    y_train: 训练集标签
    cv: 交叉验证折数
    scoring: 评分标准
    n_jobs: 并行作业数
    verbose: 详细程度
    random: 是否使用随机搜索
    n_iter: 如果使用随机搜索，设置迭代次数
    
    返回:
    best_model: 调优后的最佳模型
    search: 网格搜索或随机搜索对象
    """
    # 创建交叉验证对象
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 设置并行作业数（如果默认为-1，则使用CPU核心数-1的值）
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"使用 {n_jobs} 个CPU核心进行并行计算")
    
    # 自定义的回调函数，使用tqdm显示进度
    def fit_and_log(estimator, X, y, param_grid, total_iters, fit_params=None):
        """使用进度条跟踪拟合过程"""
        params_list = list(param_grid) if isinstance(param_grid, dict) else list(param_grid)
        progress_bar = tqdm(total=total_iters, desc="参数搜索进度", unit="fit")
        original_fit = estimator.fit
        
        def fit_with_progress(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            progress_bar.update(1)
            return result
        
        estimator.fit = fit_with_progress
        return estimator
    
    # 计算总共需要拟合的次数
    if random:
        total_fits = n_iter * cv
    else:
        # 计算网格搜索的参数组合数
        n_combinations = 1
        for param_values in param_grid.values():
            n_combinations *= len(param_values)
        total_fits = n_combinations * cv
    
    print(f"总共需要进行 {total_fits} 次拟合，可能需要较长时间...")
    
    # 根据是否使用随机搜索选择搜索方法
    if random:
        print(f"使用随机搜索优化模型参数 (n_iter={n_iter})...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=skf,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42,
            return_train_score=True
        )
    else:
        print("使用网格搜索优化模型参数...")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=skf,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用带进度条的拟合过程
    print("\n开始拟合过程...")
    
    # 创建总进度条
    with tqdm(total=100, desc="总体进度", unit="%", position=0) as pbar:
        # 执行参数搜索
        search.fit(X_train, y_train)
        pbar.update(100)  # 完成后更新到100%
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 获取最佳模型
    best_model = search.best_estimator_
    
    # 输出结果
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒" if hours > 0 else f"{int(minutes)}分钟{seconds:.2f}秒"
    
    print(f"\n参数优化完成！耗时: {time_str}")
    print(f"最佳参数: {search.best_params_}")
    print(f"最佳交叉验证分数: {search.best_score_:.4f}")
    
    # 如果有多个评分标准，打印所有评分结果
    if hasattr(search, 'multimetric_') and search.multimetric_:
        print("\n所有评分结果:")
        for metric_name, score in search.cv_results_['mean_test_score'].items():
            print(f"{metric_name}: {score:.4f}")
    
    return best_model, search

def reduce_param_grid_size(param_grid, max_combinations=100):
    """
    减少参数网格的大小，以加快搜索速度
    
    参数:
    param_grid: 原始参数网格
    max_combinations: 最大允许的组合数
    
    返回:
    reduced_param_grid: 减小后的参数网格
    """
    # 计算当前组合数
    current_combinations = 1
    for values in param_grid.values():
        current_combinations *= len(values)
    
    # 如果当前组合数小于最大允许值，直接返回原始网格
    if current_combinations <= max_combinations:
        return param_grid
    
    # 复制原始参数网格
    reduced_grid = param_grid.copy()
    
    # 计算需要减少的比例
    reduction_factor = (max_combinations / current_combinations) ** (1 / len(param_grid))
    
    # 对每个参数进行采样以减少组合数
    for param, values in reduced_grid.items():
        if isinstance(values, list) and len(values) > 1:
            # 计算减少后的参数数量
            new_size = max(2, int(len(values) * reduction_factor))
            if new_size < len(values):
                # 等间隔采样
                step = len(values) // new_size
                reduced_grid[param] = values[::step][:new_size]
    
    # 计算新的组合数
    new_combinations = 1
    for values in reduced_grid.values():
        new_combinations *= len(values)
    
    print(f"参数网格已优化: 从 {current_combinations} 种组合减少到 {new_combinations} 种组合")
    return reduced_grid

def plot_search_results(search, output_dir=None):
    """
    绘制参数搜索结果
    
    参数:
    search: GridSearchCV 或 RandomizedSearchCV 对象
    output_dir: 输出目录
    """
    try:
        # 获取结果并转换为DataFrame
        results = pd.DataFrame(search.cv_results_)
        
        print("正在绘制参数搜索结果...")
        
        # 绘制训练分数和验证分数
        plt.figure(figsize=(12, 6))
        plt.plot(results['mean_train_score'], 'o-', label='训练分数')
        plt.plot(results['mean_test_score'], 'o-', label='验证分数')
        plt.xlabel('参数组合索引')
        plt.ylabel('分数')
        plt.title('参数组合的训练和验证分数')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_path = os.path.join(output_dir, 'parameter_search_results.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存图片至: {save_path}")
        
        # 设置非阻塞模式显示图形
        plt.ion()
        plt.show()
        plt.pause(1)
        
        # 绘制前十名参数组合的分数
        top_results = results.sort_values('mean_test_score', ascending=False).head(10)
        
        plt.figure(figsize=(14, 6))
        bar_plot = plt.bar(range(len(top_results)), top_results['mean_test_score'], yerr=top_results['std_test_score'])
        
        # 在柱状图上方添加数值标签
        for i, bar in enumerate(bar_plot):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{top_results["mean_test_score"].iloc[i]:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('参数组合排名')
        plt.ylabel('验证分数')
        plt.title('前十名参数组合的验证分数')
        plt.xticks(range(len(top_results)), [f"组合 {i+1}" for i in range(len(top_results))])
        plt.grid(axis='y')
        
        # 保存图片
        if output_dir:
            save_path = os.path.join(output_dir, 'top_parameter_combinations.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存图片至: {save_path}")
        
        plt.show()
        plt.pause(1)
        plt.ioff()
        
    except Exception as e:
        print(f"绘图过程中出错: {str(e)}")
        traceback.print_exc()

def evaluate_model(model, X_test, y_test, output_dir=None):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试集特征
    y_test: 测试集标签
    output_dir: 输出目录
    
    返回:
    metrics: 包含各种评估指标的字典
    """
    # 预测类别和概率
    y_pred = model.predict(X_test)
    
    # 如果模型支持概率预测
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        has_proba = False
    
    # 计算各种评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred)
    }
    
    # 如果支持概率预测，计算AUC
    if has_proba:
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # 输出结果
    print("\n模型评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 输出分类报告
    print("\n详细分类报告:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    
    # 保存分类报告到文件
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_name = type(model).__name__
        report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型评估结果:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n详细分类报告:\n")
            f.write(report)
        
        print(f"分类报告已保存至: {report_path}")
    
    # 绘制混淆矩阵（使用matplotlib而非seaborn）
    try:
        cm = confusion_matrix(y_test, y_pred)
        
        # 获取分类数
        n_classes = cm.shape[0]
        
        # 创建图形
        plt.figure(figsize=(8, 6))
        
        # 创建混淆矩阵的热图
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        # 设置坐标轴标签
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, range(n_classes))
        plt.yticks(tick_marks, range(n_classes))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在矩阵中添加文本标注
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        # 保存混淆矩阵图
        if output_dir:
            model_name = type(model).__name__
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {cm_path}")
        
        plt.show()
        plt.pause(1)
    except Exception as e:
        print(f"绘制混淆矩阵出错: {str(e)}")
        traceback.print_exc()
    
    # 绘制评估结果的条形图
    try:
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color='skyblue')
        
        # 在柱状图上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.ylim(0, 1.1)
        plt.title('模型评估指标')
        plt.xlabel('评估指标')
        plt.ylabel('分数')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图片
        if output_dir:
            save_path = os.path.join(output_dir, f'{model_name}_evaluation_metrics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存评估指标图至: {save_path}")
        
        plt.show()
        plt.pause(1)
    except Exception as e:
        print(f"绘制评估指标图出错: {str(e)}")
    
    # 如果支持概率预测，绘制ROC曲线
    if has_proba:
        try:
            from sklearn.metrics import roc_curve, auc
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # 绘制ROC曲线
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率')
            plt.ylabel('真正例率')
            plt.title('接收者操作特征曲线 (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # 保存ROC曲线图
            if output_dir:
                roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                print(f"ROC曲线已保存至: {roc_path}")
            
            plt.show()
            plt.pause(1)
        except Exception as e:
            print(f"绘制ROC曲线出错: {str(e)}")
    
    return metrics

def save_model(model, model_name, dataset_name, output_dir):
    """
    保存模型及其元数据
    
    参数:
    model: 训练好的模型
    model_name: 模型名称
    dataset_name: 数据集名称
    output_dir: 输出目录
    
    返回:
    model_path: 模型保存路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建模型文件名
    model_filename = f"{model_name}_{dataset_name}_tuned_model.pkl"
    model_path = os.path.join(output_dir, model_filename)
    
    # 保存模型
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")
    
    return model_path

def tune_and_evaluate(data_path, model_name, output_dir=None, random_search=False, n_iter=10, cv=5, scoring='f1', save=True, reduce_params=False):
    """
    完整的模型调优和评估流程
    
    参数:
    data_path: 标准化数据集的路径
    model_name: 模型名称，如'rf', 'svm', 'lr', 'knn', 'dt', 'gb'
    output_dir: 输出目录
    random_search: 是否使用随机搜索
    n_iter: 随机搜索迭代次数
    cv: 交叉验证折数
    scoring: 评分标准
    save: 是否保存模型
    reduce_params: 是否减少参数组合数以加快训练
    
    返回:
    best_model: 调优后的最佳模型
    metrics: 评估指标
    """
    print("\n" + "="*50)
    print(f"开始处理数据集: {data_path}")
    print(f"模型: {model_name}, 评分标准: {scoring}, CV: {cv}")
    print("="*50 + "\n")
    
    # 1. 加载数据
    data = load_standardized_data(data_path)
    if data is None:
        print("数据加载失败，终止处理")
        return None, None
    
    # 提取数据集名称
    dataset_name = os.path.basename(os.path.dirname(data_path))
    print(f"数据集名称: {dataset_name}")
    
    # 2. 获取模型和参数网格
    try:
        print(f"获取模型 '{model_name}' 及其参数网格...")
        model, param_grid = get_model_and_param_grid(model_name, reduce_params)
        print(f"参数网格大小: {len(param_grid)}")
    except ValueError as e:
        print(str(e))
        return None, None
    
    # 3. 参数调优
    print(f"开始参数调优过程，使用{'随机搜索' if random_search else '网格搜索'}...")
    best_model, search = tune_model_parameters(
        model, 
        param_grid, 
        data['X_train'], 
        data['y_train'], 
        cv=cv,
        scoring=scoring,
        random=random_search,
        n_iter=n_iter
    )
    
    # 4. 绘制参数搜索结果
    model_output_dir = None
    if output_dir:
        model_output_dir = os.path.join(output_dir, dataset_name, model_name)
        print(f"保存可视化结果到: {model_output_dir}")
        plot_search_results(search, model_output_dir)
    else:
        print("不保存可视化结果，仅显示")
        plot_search_results(search)
    
    # 5. 在测试集上评估模型
    metrics = None
    if data['X_test'] is not None and data['y_test'] is not None:
        print("在测试集上评估最佳模型...")
        metrics = evaluate_model(best_model, data['X_test'], data['y_test'], model_output_dir)
    else:
        print("没有测试集数据，跳过评估步骤")
    
    # 6. 保存模型
    if save and output_dir:
        model_output_dir = os.path.join(output_dir, dataset_name, model_name)
        print(f"保存模型到: {model_output_dir}")
        save_model(best_model, model_name, dataset_name, model_output_dir)
    else:
        print("不保存模型")
    
    print("\n" + "="*50)
    print("处理完成")
    print("="*50 + "\n")
    
    return best_model, metrics

def main():
    """
    主函数
    """
    import argparse
    
    print("\n开始执行模型调参脚本...")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='使用交叉验证和网格搜索为模型调参')
    parser.add_argument('--input', type=str, help='输入的标准化数据集NPZ文件路径', required=True)
    parser.add_argument('--model', type=str, help='模型名称: rf, svm, lr, knn, dt, gb', required=True)
    parser.add_argument('--output', type=str, help='输出目录', default='NASADefectDataset/Models')
    parser.add_argument('--random', action='store_true', help='使用随机搜索而非网格搜索')
    parser.add_argument('--n_iter', type=int, help='随机搜索迭代次数', default=10)
    parser.add_argument('--cv', type=int, help='交叉验证折数', default=5)
    parser.add_argument('--scoring', type=str, help='评分标准', default='f1')
    parser.add_argument('--no_save', action='store_true', help='不保存模型')
    parser.add_argument('--reduce_params', action='store_true', help='减少参数组合数以加快训练')
    
    args = parser.parse_args()
    
    print(f"脚本运行于目录: {os.getcwd()}")
    print(f"参数: {args}")
    
    try:
        tune_and_evaluate(
            args.input,
            args.model,
            args.output,
            random_search=args.random,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            save=not args.no_save,
            reduce_params=args.reduce_params
        )
    except Exception as e:
        print(f"执行过程中发生异常: {str(e)}")
        print("详细异常信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main()