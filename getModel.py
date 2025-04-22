from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelInitializer:
    """模型初始化类，用于初始化和配置各种机器学习模型"""
    
    def __init__(self):
        """初始化ModelInitializer类"""
        self.models = {}
        self.initialize_all_models()
    
    def initialize_all_models(self):
        """初始化所有默认模型"""
        self.initialize_knn()
        self.initialize_naive_bayes()
        self.initialize_decision_tree()
        self.initialize_random_forest()
    
    def initialize_knn(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        """初始化KNN分类器
        
        参数:
            n_neighbors: int, 默认5, K近邻数
            weights: str, 默认'uniform', 权重类型 ('uniform' 或 'distance')
            algorithm: str, 默认'auto', 计算最近邻的算法
        """
        self.models['knn'] = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
    
    def initialize_naive_bayes(self, var_smoothing=1e-9):
        """初始化高斯朴素贝叶斯分类器
        
        参数:
            var_smoothing: float, 默认1e-9, 方差平滑参数
        """
        self.models['naive_bayes'] = GaussianNB(var_smoothing=var_smoothing)
    
    def initialize_decision_tree(
        self,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        """初始化决策树分类器
        
        参数:
            criterion: str, 默认'gini', 分割标准 ('gini' 或 'entropy')
            max_depth: int, 默认None, 树的最大深度
            min_samples_split: int, 默认2, 分裂内部节点所需的最小样本数
            min_samples_leaf: int, 默认1, 叶节点所需的最小样本数
        """
        self.models['decision_tree'] = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    
    def initialize_random_forest(
        self,
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        """初始化随机森林分类器
        
        参数:
            n_estimators: int, 默认100, 森林中树的数量
            criterion: str, 默认'gini', 分割标准 ('gini' 或 'entropy')
            max_depth: int, 默认None, 树的最大深度
            min_samples_split: int, 默认2, 分裂内部节点所需的最小样本数
            min_samples_leaf: int, 默认1, 叶节点所需的最小样本数
        """
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    
    def get_model(self, model_name):
        """获取指定的模型实例
        
        参数:
            model_name: str, 模型名称 ('knn', 'naive_bayes', 'decision_tree', 'random_forest')
        
        返回:
            指定的模型实例
        """
        return self.models.get(model_name)
    
    def get_all_models(self):
        """获取所有模型实例
        
        返回:
            包含所有模型的字典
        """
        return self.models