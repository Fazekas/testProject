from sklearn.datasets import fetch_openml
import numpy as np
from numpy import loadtxt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# just to make things better. There are 60000 training samples and 10000 test samples.
# use this number to get the first x amount and then get the remaining after x
mnist_data_split = 60000


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:mnist_data_split] = mnist.data[reorder_train]
    mnist.target[:mnist_data_split] = mnist.target[reorder_train]
    mnist.data[mnist_data_split:] = mnist.data[reorder_test + mnist_data_split]
    mnist.target[mnist_data_split:] = mnist.target[reorder_test + mnist_data_split]


print('Fetching mnist data')
mnist = fetch_openml('mnist_784', version=1, cache=True)
print('Got mnist data')
mnist.target = mnist.target.astype(np.int8)  # returns targets as strings
sort_by_target(mnist)  # sorts the data set
print(mnist)
X, y = mnist["data"], mnist["target"]

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')
plt.show()

X_train, X_test = X[:mnist_data_split], X[mnist_data_split:]
y_train, y_test = y[:mnist_data_split], y[mnist_data_split:]

# X_train, X_test = X_train.astype(int), X_test.astype(int)
# y_train, y_test = y_train.astype(int), y_test.astype(int)

# shuffle everything to that theres a somewhat even spread between tests and training sets
shuffle_index = np.random.permutation(mnist_data_split)

try:
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
except:
    print('An exception was thrown with the large data set. Reading from the files instead')
    X_train = loadtxt('X_train.csv', delimiter=',')
    y_train = loadtxt('y_train.csv', delimiter=',')

#
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
# Fit the model with stochastic Gradient Descent.
# Place the target values in with the training set
# to be able to make a prediction on a random value on if it matches the target values
sgd_clf.fit(X_train, y_train_5)

# Make a prediction that some_digit is a 5
sgd_clf.predict([some_digit])

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))



