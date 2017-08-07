# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

plt.figure(figsize=(17, 3))

x = np.arange(-5, 5, 0.001)
y1 = np.tanh(x)
y2 = 1 / (1 + np.exp(-x))
y3 = np.zeros(x.shape)
y3 = np.maximum(x, 0)
y4 = []
for i in x:
    if i < 0:
        y4.append(0.05 * i)
    else:
        y4.append(i)

plt.subplot(141)
plt.plot(x, y1)
plt.plot(np.arange(-5, 6), np.zeros(11), 'black', alpha=0.5)
plt.plot(np.zeros(3), np.arange(-1, 2), 'black', alpha=0.5)
plt.title("$Tanh ~ (-1, 1)$", size=15)
plt.grid(True)
plt.subplot(142)
plt.plot(x, y2)
plt.plot(np.arange(-5, 6), np.zeros(11), 'black', alpha=0.5)
plt.plot(np.zeros(2), np.arange(0, 2), 'black', alpha=0.5)
plt.title("$Sigmoid ~ (0, 1)$", size=15)
plt.grid(True)
plt.subplot(143)
plt.plot(x, y3)
plt.title("$\mathrm{ReLU}$", size=15)
plt.ylim([-0.5, 5])
plt.grid(True)
plt.subplot(144)
plt.plot(x, y4)
plt.title("$\mathrm{Leaky ~ ReLU}$", size=15)
plt.ylim([-0.5, 5])
plt.grid(True)
plt.show()