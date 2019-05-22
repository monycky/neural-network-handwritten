from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2

dataset = fetch_openml('mnist_784', version=1, cache=True)
x_train, y_train = mnist.data / 255., mnist.target

#Features
X = np.array(dataset.data)

#Labels
y = np.array(dataset.target)

X =  X.astype('float32') 

#Splitting Dataset into Training and Testing dataset
#First 60k instances are for Training and last 10k are for testing
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

#Normalizing Our Features in gray scale
X_train = X_train /255
X_test = X_test /255

mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(50), activation='relu',learning_rate='adaptive')

#fitting the model
mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train)) 
print("Test set score: %f" % mlp.score(X_test, y_test))

confusion_matrix(y_test,mlp.predict(X_test))
classification_report(y_test,mlp.predict(X_test))

for i in range(10):
    img = cv2.imread('./images/%d.png' % i, 0)
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res = mlp.reshape((28 * 28))

    res = res / 255.
    print(i, mlp.predict([res]))
