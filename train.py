import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from getModel import ModelInitializer

class ModelTrainer:
    """模型训练类，使用最佳参数训练模型并生成评估报告"""
    
    def __init__(self):
        """初始化ModelTrainer类"""
        self.model_initializer = ModelInitializer()
        self.optimization_dir = os.path.join(os.path.dirname(__file__), 'OptimizationResults')
        self.dataset_dir = os.path.join(os.path.dirname(__file__), 'StandardizedDatasets')
        
    def load_best_params(self, project, model_name):
        """加载指定项目的最佳参数"""
        param_file = os.path.join(self.optimization_dir, f"{project}_{model_name}_best_params.json")
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"未找到参数文件: {param_file}")
            
        with open(param_file, 'r') as f:
            return json.load(f)
    
    def train_model(self, model_name, params, X_train, y_train):
        """使用最佳参数训练模型"""
        model = self.model_initializer.get_model(model_name)
        if model is None:
            raise ValueError(f"未找到模型: {model_name}")
            
        model.set_params(**params)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能并生成报告"""
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()
        
    def save_report(self, report, project, model_name):
        """保存评估报告到EvaluationResults目录"""
        eval_dir = os.path.join(os.path.dirname(__file__), 'EvaluationResults')
        os.makedirs(eval_dir, exist_ok=True)
        
        report_file = os.path.join(eval_dir, f"{project}_{model_name}_report.csv")
        report.to_csv(report_file, index=True)
        print(f"评估报告已保存到: {report_file}")
    
    def train_and_evaluate_all(self):
        """训练并评估所有模型"""
        projects = ['apache', 'safe', 'zxing']
        models = ['knn', 'naive_bayes', 'decision_tree', 'random_forest']
        
        for project in projects:
            print(f"\n正在处理项目: {project}")
            
            # 加载测试数据
            X_test = pd.read_csv(os.path.join(self.dataset_dir, project, 'x_test.csv'))
            y_test = pd.read_csv(os.path.join(self.dataset_dir, project, 'y_test.csv')).values.ravel()
            
            # 加载训练数据
            X_train = pd.read_csv(os.path.join(self.dataset_dir, project, 'x_train.csv'))
            y_train = pd.read_csv(os.path.join(self.dataset_dir, project, 'y_train.csv')).values.ravel()
            
            for model_name in models:
                try:
                    # 加载最佳参数
                    best_params = self.load_best_params(project, model_name)
                    
                    # 训练模型
                    model = self.train_model(model_name, best_params, X_train, y_train)
                    
                    # 评估模型
                    report = self.evaluate_model(model, X_test, y_test)
                    
                    print(f"\n{model_name} 模型评估报告:")
                    print(report)
                    self.save_report(report, project, model_name)
                    
                except Exception as e:
                    print(f"\n{model_name} 模型训练或评估失败: {str(e)}")

def main():
    """主函数"""
    trainer = ModelTrainer()
    trainer.train_and_evaluate_all()

if __name__ == "__main__":
    main()