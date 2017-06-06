# coding=utf-8
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

data = pd.read_csv("census.csv")
income_raw = data['income']                                 # label column: 1
features_raw = data.drop('income', axis=1)                  # feature column: 13

# 数值型特征预处理：特征缩放
# 对于符合长尾分布（倾斜）的特征，用对数变换压缩X轴取值范围
# 原数据中，capital-gain 取值范围为 [0, 99999], capital-loss 则为 [0, 4356]
# 压缩后的数据中，capital-gain 取值范围为 [0, 11], capital-loss 则为 [0, 8]
skewed = ['capital-gain', 'capital-loss']
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 字符型特征预处理：One-Hot Encoding
# 必须要确保特征数据的元素是可以运算的类型，例如数值型或者布尔型。Object和String不行。
features = pd.get_dummies(features_raw)
print "{} total features after one-hot encoding.".format(features.shape[1])

# 字符型特征的另一种预处理方式：Label Encoding
cat_features = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
num_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
trans_features = {}
for f in cat_features:
    labels = data[f].values
    le = LabelEncoder()
    le.fit(labels)
    trans = le.transform(data[f])
    trans_features[f] = pd.Series(trans)
for f in num_features:
    trans_features[f] = pd.Series(features[f])
trans_features = pd.DataFrame(trans_features)

# 标签预处理：直接赋值为0和1
income = (income_raw == '>50K').astype(int)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0)
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(trans_features, income, test_size=0.2, random_state=0)

# 训练1：使用One-Hot Encoder的训练数据
clf = DecisionTreeClassifier(random_state=0, max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
imp = clf.feature_importances_
indices = np.argsort(imp)[::-1]                 # Sort according to importance
columns = X_train.columns.values[indices[:5]]   # Select the first 5 features
values = imp[indices][:5]                       # Get the corresponding importance
print "\nThe Most Important 5 Features: "
for i in range(columns.shape[0]):
    print '-', columns[i], '=', values[i]
print 'Accuracy =', accuracy_score(y_test, y_pred)
print 'F0.5 =', fbeta_score(y_test, y_pred, 0.5)

# 训练2：使用LabelEncoder的训练数据
clf_trans = DecisionTreeClassifier(random_state=0, max_depth=5)
clf_trans.fit(X_train_trans, y_train_trans)
y_pred_trans = clf_trans.predict(X_test_trans)
imp_trans = clf_trans.feature_importances_
indices = np.argsort(imp_trans)[::-1]                 # Sort according to importance
columns = X_train_trans.columns.values[indices[:5]]   # Select the first 5 features
values = imp_trans[indices][:5]                       # Get the corresponding importance
print "\nThe Most Important 5 Features: "
for i in range(columns.shape[0]):
    print '-', columns[i], '=', values[i]
print 'Accuracy =', accuracy_score(y_test_trans, y_pred_trans)
print 'F0.5 =', fbeta_score(y_test_trans, y_pred_trans, 0.5)
export_graphviz(clf_trans, out_file='decisiontree.dot')

def get_importance(importances, name):
    feature_name = np.where(X_train.columns.str.contains(name))
    print name, "importance:", np.sum(importances[feature_name])




