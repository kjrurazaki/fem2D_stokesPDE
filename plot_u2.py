import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

Z = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) - np.sin(2 * np.pi * X)

plt.contourf(X, Y, Z, cmap="viridis", levels=100)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="sin(2πx) cos(2πy) - sin(2πx)")
plt.title("Contour plot of sin(2πx) cos(2πy) - sin(2πx)")
plt.show()
