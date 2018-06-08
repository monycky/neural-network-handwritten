from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import cv2

mnist = fetch_mldata("MNIST original")
x_train, y_train = mnist.data / 255., mnist.target

clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30, 30))

clf.fit(x_train, y_train)

for i in range(10):
    img = cv2.imread('C:/Users/monycky_vasconcelos/Documents/projects/ann-handwritten/images/%d.png' % i, 0)
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res = res.reshape((28 * 28))

    res = res / 255.
    print(i, clf.predict([res]))
