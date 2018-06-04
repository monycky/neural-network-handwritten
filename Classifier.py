from sklearn.neural_network import MLPClassifier


class Classifier:
    x_train = [[2.0, 3.0], [1.0, 1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]
    y_train = [5.0, 2.0, 0.0, 3.0, 6.0]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-3,
                        hidden_layer_sizes=5, random_state=1)

    clf.fit(x_train, y_train)

    print(clf.predict([[3.0, 1.0]]))
