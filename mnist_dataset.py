print(__doc__)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")

x, y = mnist.data / 255., mnist.target

for i in range(10000)
    data = (x[10000])
    label = (y[10000])
pixels = data.reshape((28, 28))


print(data)
print("label", label)
plt.imshow(pixels, cmap="gray")
plt.show()
