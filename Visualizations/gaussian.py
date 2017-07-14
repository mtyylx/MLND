import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10000)            # x is only index
y = np.random.randn(10000)      # y is not a function of x

plt.figure(figsize=(18, 5))
plt.suptitle("     Gaussian Random Variable", size=18)

plt.subplot(131)
plt.scatter(x, y, s=5, alpha=0.5)
plt.title("Value Distribution", size=15)
plt.xlabel("Sample Index")
plt.ylabel("Sample Value")

plt.subplot(132)
plt.hist(y, bins=99)
plt.title("Value Histogram (PDF)", size=15)
plt.xlabel("Sample Value")
plt.ylabel("Value Occurrence Count")

plt.subplot(133)
x = np.arange(-5, 5, 0.01)
y = 1.0/2 * np.pi * np.exp(-(x/2.0)**2)               # y is an exponential function of x
plt.plot(x, y)
plt.title("Probability Density Curve (PDF)", size=15)
plt.xlabel("Density Input")
plt.ylabel("Density Value")
plt.show()