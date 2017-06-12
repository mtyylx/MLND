import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)

x = np.arange(0, 8, 0.001).reshape((8000, 1))
y0 = np.cos(x)
y1 = 1 - x**2/2
y2 = y1 + x**4/24
y3 = y2 - x**6/720
y4 = y3 + x**8/40320
y5 = y4 - x**10/3628800
y6 = y5 + x**12/479001600
y7 = y6 - x**14/87178291200

plt.figure(figsize=(10,6))
plt.title('$\mathbf{Use ~ Polynomial ~ Function ~ to ~ Fit ~ Cosine}$', size=20)
f0 = plt.plot(x, y0, label='$\cos{x}$', linewidth=3, linestyle='--', color='crimson')
f1 = plt.plot(x, y1, label='$O(x^2)$', color='C1')
f2 = plt.plot(x, y2, label='$O(x^4)$', color='C2')
f3 = plt.plot(x, y3, label='$O(x^6)$', color='C9')
f4 = plt.plot(x, y4, label='$O(x^8)$', color='darkcyan')
f5 = plt.plot(x, y5, label='$O(x^{10})$', color='purple')
f6 = plt.plot(x, y6, label='$O(x^{12})$', color='blue')
f7 = plt.plot(x, y7, label='$O(x^{14})$', color='midnightblue')
plt.xlabel('$x$', size=15)
plt.ylim([-3, 3])
plt.grid(True)
plt.legend(fontsize='13')
plt.show()
plt.savefig('series')