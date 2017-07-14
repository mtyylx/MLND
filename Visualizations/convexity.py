import matplotlib.pyplot as plt
import numpy as np


a = np.arange(-4 * np.pi, 4 * np.pi, 0.001)
convex = a**2
concave = -a**2
nonconvex = a**2 + 20 * np.sin(2*a)
plt.figure(figsize=(17,5))
plt.subplot(131)
plt.plot(a, convex, color='r')
plt.title('Convex', size=20)
plt.grid(True)
plt.subplot(132)
plt.plot(a, concave, color = 'g')
plt.title('Concave', size=20)
plt.grid(True)
plt.subplot(133)
plt.plot(a, nonconvex, color = 'b')
plt.title('Non-Convex', size=20)
plt.grid(True)
plt.show()