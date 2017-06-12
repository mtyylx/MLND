# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

x = np.arange(0, 10).reshape((10, 1))
# Truth (But in reality unknown to us)
y = x * 0.5

# Prediction (Univariate Linear Regression)
w = np.arange(-1, 2, 0.1).reshape((30, 1))
# np.sum 按行求和
# np.multiply 矩阵乘法
# x.T 列向量转置
# **2 按每个元素求平方
mse = np.sum((np.multiply(w, x.T) - y.T)**2, axis = 1)

plt.figure(figsize=(15, 5))

# Hypothesis
plt.subplot(121)
plt.title('$Hypothesis ~ Function ~ \hat{y}(x)$', size=15)
w_sample = np.arange(-0.5, 2, 0.5).reshape((5, 1))
y_pred = np.multiply(w_sample, x.T)
plt.xlabel('$x$', size=15)
plt.ylabel('$\hat{y}$', size=15, rotation=0)
plt.scatter(x, y_pred[0], color='purple')
plt.scatter(x, y_pred[1], color='orange')
plt.scatter(x, y_pred[2], color='red', marker='x')
plt.scatter(x, y_pred[3], color='green')
plt.scatter(x, y_pred[4], color='blue')
reg = plt.plot(x, y, 'r')
plt.annotate('$w = 1.5$', xy=(1, 3), xytext=(6.91, 11.36), size=10)
plt.annotate('$w = 1.0$', xy=(1, 3), xytext=(6.91, 8.25), size=10)
plt.annotate('$w = 0.5$', xy=(1, 3), xytext=(6.91, 4.28), size=10)
plt.annotate('$w = 0.0$', xy=(1, 3), xytext=(6.91, 1.0), size=10)
plt.annotate('$w = -0.5$', xy=(1, 3), xytext=(6.80, -2.2), size=10)
plt.legend(reg, ('$\hat{y}(x) = wx$',), fontsize=20)
plt.grid(True)

# Cost Function
plt.subplot(122)
plt.title('$Cost ~ Function ~ J(w)$', size=15)
reg2 = plt.plot(w, mse)
mse_sample = range(5, 30, 5)
plt.xlabel('$w$', size=15)
plt.ylabel('$J$', size=15, rotation=0)
plt.scatter(-0.5, mse[5], color='purple', s=100)
plt.scatter( 0.0, mse[10], color='orange', s=100)
plt.scatter( 0.5, mse[15], color='red', s=100)
plt.scatter( 1.0, mse[20], color='green', s=100)
plt.scatter( 1.5, mse[25], color='blue', s=100)
plt.plot(0.5 * np.ones((10, 1)), np.arange(-100, 900, 100), color='black', linestyle='dotted')
plt.plot(np.arange(-5, 5, 1), np.zeros((10, 1)), color='black')
plt.annotate('$Slope < 0 \leftarrow$', xy=(1, 3), xytext=(0.05, 347), size=10)
plt.annotate('$\\rightarrow Slope > 0 $', xy=(1, 3), xytext=(0.498, 347), size=10)
plt.legend(reg2, ('$J(w) = \sum_{i=1}^m {\left(wx^{(i)} - y^{(i)}\\right)^2}$',), fontsize=20)
plt.xlim([-1.1, 2])
plt.ylim([-40, 670])
plt.grid(True)
plt.show()