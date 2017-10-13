import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(6)]
y = [i**2 for i in range(6)]

plt.plot(x, y)

plt.xlabel('i')
plt.ylabel('i^2')

plt.grid(True)

plt.savefig("test.png")

plt.show()