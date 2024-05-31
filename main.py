import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据集准备
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 数据集切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.237, random_state=84)

# 数据预处理: 将数据规格化到[0,1]之间
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))


# 训练高斯朴素贝叶斯分类器
class Bayes:
    def __init__(self, var_smoothing=1e-9):  # 添加平滑参数
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_count = len(self.classes)
        self.mean = []
        self.var = []
        self.prior = []  # 添加先验概率列表
        for c in self.classes:
            X_class = X[y == c]
            self.mean.append(np.mean(X_class, axis=0))
            self.var.append(np.var(X_class, axis=0) + self.var_smoothing)  # 加上平滑值
            self.prior.append(len(X_class) / len(X))  # 计算先验概率

    def predict(self, X):
        y_pred = []
        for x in X:
            # 使用高斯分布的概率密度函数和先验概率计算后验概率，取对数避免下溢
            log_prob = [
                np.log(self.prior[c]) - 0.5 * np.sum(np.log(2 * np.pi * self.var[c])) -
                0.5 * np.sum((x - self.mean[c]) ** 2 / self.var[c])
                for c in range(self.class_count)]
            y_pred.append(self.classes[np.argmax(log_prob)])  # 选择最大后验概率对应的类别
        return y_pred


# 训练模型并测试准确率
gnb = Bayes()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
