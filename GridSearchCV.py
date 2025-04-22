import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from getModel import ModelInitializer
from sklearn.model_selection import GridSearchCV, KFold
import warnings
from sklearn.exceptions import DataConversionWarning

class ModelOptimizer:
    """模型优化类，使用网格搜索和交叉验证进行参数优化"""
    
    def __init__(self):
        """初始化ModelOptimizer类"""
        self.model_initializer = ModelInitializer()
        self.param_grids = self._initialize_param_grids()
        self.scoring = self._initialize_scoring()
    
    def _initialize_param_grids(self):
        """初始化各个模型的参数网格"""
        return {
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'naive_bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'decision_tree': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    
    def _initialize_scoring(self):
        """初始化评估指标"""
        return {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0)
        }
    
    def optimize_model(self, model_name, X_train, y_train, cv=10):
        """对指定模型进行参数优化
        
        参数:
            model_name: str, 模型名称
            X_train: array-like, 训练数据特征
            y_train: array-like, 训练数据标签
            cv: int, 默认5, 交叉验证折数
        
        返回:
            dict: 包含最佳参数和评估结果的字典
        """
        # 获取模型实例
        model = self.model_initializer.get_model(model_name)
        if model is None:
            raise ValueError(f"未找到模型: {model_name}")
        
        # 获取参数网格
        param_grid = self.param_grids.get(model_name)
        if param_grid is None:
            raise ValueError(f"未找到模型的参数网格: {model_name}")
        
        # 创建GridSearchCV实例
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1,
            refit='f1'  # 使用F1分数作为最终模型选择标准
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 返回优化结果
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
    
    def optimize_all_models(self, X_train, y_train, cv=10):
        """优化所有模型的参数
        
        参数:
            X_train: array-like, 训练数据特征
            y_train: array-like, 训练数据标签
            cv: int, 默认5, 交叉验证折数
        
        返回:
            dict: 包含所有模型优化结果的字典
        """
        results = {}
        for model_name in self.param_grids.keys():
            try:
                results[model_name] = self.optimize_model(model_name, X_train, y_train, cv)
                print(f"\n{model_name} 模型优化完成:")
                print(f"最佳参数: {results[model_name]['best_params']}")
                print(f"最佳F1分数: {results[model_name]['best_score']:.4f}")
            except Exception as e:
                print(f"\n{model_name} 模型优化失败: {str(e)}")
        return results
        

def main():
    """主函数，用于加载数据并执行模型优化"""
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 初始化模型优化器
    optimizer = ModelOptimizer()
    # 初始化模型

    # 加载标准化数据集
    dataset_dir = os.path.join(os.path.dirname(__file__), 'StandardizedDatasets')
    projects = ['apache', 'safe', 'zxing']
    
    for project in projects:
        print(f"\n正在处理项目: {project}")

        # 加载训练数据
        x_train_file = os.path.join(dataset_dir, project, 'x_train.csv')
        x_train_data = pd.read_csv(x_train_file)
        
        y_train_file = os.path.join(dataset_dir, project, 'y_train.csv')
        y_train_data = pd.read_csv(y_train_file).values.ravel()
        
        
        # 执行模型优化
        results = optimizer.optimize_all_models(x_train_data, y_train_data)
        
        # 保存结果
        output_dir = os.path.join(os.path.dirname(__file__), 'OptimizationResults')
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, result in results.items():
            result_file = os.path.join(output_dir, f"{project}_{model_name}_best_params.json")
            with open(result_file, 'w') as f:
                json.dump(result['best_params'], f)
            print(f"{model_name} 最佳参数已保存到: {result_file}")

if __name__ == "__main__":
    main()
        
    