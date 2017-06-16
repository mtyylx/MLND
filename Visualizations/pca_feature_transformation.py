import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

x = np.arange(1, 9, 0.1)
y = 0.5 * x + np.random.rand(x.shape[0]) - 0.5
plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=10)
plt.xlim([-2, 12])
plt.ylim([-1, 6])
plt.title('$\mathbf{Principal ~Components ~Analysis}$', size=15)
plt.annotate('$x^{\prime}_1$', xy=(1, 3), xytext=(9.40, 4.41), size=15)
plt.annotate('$x^{\prime}_2$', xy=(1, 3), xytext=(0.74, 5), size=15)
plt.annotate('$x_1$', xy=(1, 3), xytext=(10.28, -0.4), size=15)
plt.annotate('$x_2$', xy=(0, 0), xytext=(-0.64, 5.2), size=15)
ax = plt.axes()
ax.arrow(0, 0, 10, 0, linewidth=2, head_width=0.15, head_length=0.3, fc='k', ec='k')
ax.arrow(0, 0, 0, 5, linewidth=2, head_width=0.15, head_length=0.3, fc='k', ec='k')
ax.arrow(1, 0.5, 8, 4, linewidth=2, head_width=0.15, head_length=0.3, fc='red', ec='red')
ax.arrow(3, 0.5, -2, 4, linewidth=2, head_width=0.15, head_length=0.3, fc='red', ec='red')
plt.grid(True)
plt.show()