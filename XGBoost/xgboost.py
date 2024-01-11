import numpy as np
import pandas as pd
from typing import Set, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


iris_data = pd.read_csv('D:\\Users\\my_projects\\vscode_projects\\deep_learning\\datasets/iris.data')
le = LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
iris_data['class'] = le.transform(iris_data['class'])

features = iris_data.columns.to_list()
features.remove('class')

train_x, test_x, train_y, test_y = train_test_split(iris_data[features], iris_data['class'],
                                                    test_size=0.2, shuffle=True, random_state=2023)


class XGBoost:
    def __init__(self, max_depth=3, learning_rate=0.1, reg_lambda=1.0, n_estimators=100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_estimators = n_estimators
        self.trees = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, y, y_pred):
        return -2 * y * self.sigmoid(-2 * y * y_pred)

    def hessian(self, y, y_pred):
        p = self.sigmoid(y * y_pred)
        return 2 * p * (1 - p)

    def loss(self, y, y_pred):
        return np.log(1 + np.exp(-2 * y * y_pred))

    def fit(self, X, y):
        F = np.zeros(len(X))
        
        for _ in range(self.n_estimators):
            residuals = np.zeros(len(X))
            for i in range(len(X)):
                y_pred = F[i]
                g = self.gradient(y[i], y_pred)
                h = self.hessian(y[i], y_pred)
                residuals[i] = g / (h + self.reg_lambda)
            
            tree = self.build_tree(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
    
    def build_tree(self, X, residuals, depth=1):
        if depth > self.max_depth or np.abs(residuals).sum() < 1e-3:
            return None
        
        tree = DecisionTree()
        best_feature, best_threshold = tree.find_best_split(X, residuals)
        if best_feature is None:
            return None
        
        tree.feature_index = best_feature
        tree.threshold = best_threshold
        
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        tree.left = self.build_tree(X[left_indices], residuals[left_indices], depth + 1)
        tree.right = self.build_tree(X[right_indices], residuals[right_indices], depth + 1)
        
        return tree
    
    def predict(self, X):
        F = np.zeros(len(X))
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return np.sign(F)

class DecisionTree:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
    
    def find_best_split(self, X, residuals):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                
                gini = self.calculate_gini(residuals[left_indices], residuals[right_indices])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def calculate_gini(self, left_residuals, right_residuals):
        p_left = self.sigmoid(left_residuals).mean()
        p_right = self.sigmoid(right_residuals).mean()
        
        gini_left = 1 - p_left**2 - (1 - p_left)**2
        gini_right = 1 - p_right**2 - (1 - p_right)**2
        
        left_size = len(left_residuals)
        right_size = len(right_residuals)
        total_size = left_size + right_size
        
        gini = (left_size / total_size) * gini_left + (right_size / total_size) * gini_right
        return gini
    
    def predict(self, X):
        if self.feature_index is None:
            return np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        left_indices = X[:, self.feature_index] < self.threshold
        right_indices = ~left_indices
        
        if self.left is not None:
            predictions[left_indices] = self.left.predict(X[left_indices])
        if self.right is not None:
            predictions[right_indices] = self.right.predict(X[right_indices])
        
        return predictions
