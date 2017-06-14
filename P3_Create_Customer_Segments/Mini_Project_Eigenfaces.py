# coding=utf-8
from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# 使用前的准备工作：下载图像数据到用户根目录，安装PIL。

###############################################################################
# 0. 数据导入为numpy array，只选择照片资源多于70张的人物（Easy模式），每张图像尺寸都进行缩放
np.random.seed(42)
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 输入特征数据
X = lfw_people.data
n_features = X.shape[1]
n_samples, h, w = lfw_people.images.shape

# 输出数据：人物编号（照片多于70张的只有7个人） the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print "Total dataset size:"
print "样本数量 n_samples: %d" % n_samples
print "特征数量 n_features: %d" % n_features
print "分类个数 n_classes: %d" % n_classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
# 1. PCA分析：将1850个特征经过PCA分析后挑选出150个最重要的主成分，对测试集和训练集都进行PCA转换。
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

# 观察其中一人的分数随主成分数量增加的变化趋势：先提升、后退化
# Ariel Sharon's F1 Score
# 10: 0.11
# 15: 0.34
# 20: 0.40
# 25: 0.60
# 50: 0.74
# 100: 0.71
# 150: 0.72

n_components = 150
print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
t0 = time()
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True)
pca.fit(X_train)
print "done in %0.3fs" % (time() - t0)
print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.3fs" % (time() - t0)
eigenfaces = pca.components_.reshape((n_components, h, w))


###############################################################################
# 2. 将转换后的数据送入SVM进行训练
# Train a SVM classification model
print "Fitting the classifier to the training set"
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_


###############################################################################
# 3. 将模型用测试集进行性能评估（量化评估）
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0)

print 'classification report: '
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))


###############################################################################
# 4. 将模型预测进行可视化评估
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significant eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

pl.show()