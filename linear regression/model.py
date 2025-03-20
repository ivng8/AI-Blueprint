import numpy as np
import matplotlib.pyplot as plt
import math
import util

class Model:
    """
    Abstract class for a machine learning model.
    """
    
    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree = 1, learning_rate = 1e-3):
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = [1] + [0] * (degree)
 
    def get_features(self, x):
        return [x**i for i in range(self.degree + 1)]

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        curr = 0
        for w, f in zip(self.weights, self.get_features(x)):
            curr += w * f
        return curr

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        return 0.5 * (self.hypothesis(x) - y) ** 2

    def gradient(self, x, y):
        err = self.hypothesis(x) - y
        return [err * f for f in self.get_features(x)]

    def train(self, dataset, evalset = None):
        xs, ys = dataset.get_all_samples()
        eval_iters = []
        losses = []
        for i in range(10000):
            total_loss = 0
            for j in range(dataset.get_size()):
                x, y = xs[j], ys[j]
                delta_g = self.gradient(x, y)
                self.weights = [w - self.learning_rate * g for w, g in zip(self.weights, delta_g)]
                total_loss += self.loss(x, y)

            avg_loss = total_loss / dataset.get_size()

            if i % 50 == 0:
                eval_iters.append(i)
                losses.append(avg_loss)

        return eval_iters, losses


# PA4 Q2
def linear_regression():
    sine_train = util.get_dataset("sine_train")
    sine_val = util.get_dataset("sine_val")

    sine_model = PolynomialRegressionModel(1, 1e-4)
    eval_iters, losses = sine_model.train(sine_train)

    print("Final hypothesis:", sine_model.get_weights())
    train_loss = sine_train.compute_average_loss(sine_model)
    print("Average training loss:", train_loss)

    util.RegressionDataset.plot_data(sine_train, sine_model)
    sine_train.plot_loss_curve(eval_iters, losses)

    best_degree, best_lr, best_loss = 0, 0, float('inf')
    for degree in range(1, 3):
        for lr in range(-12, -3):
            model = PolynomialRegressionModel(degree, 10**lr)
            model.train(sine_train)

            train_loss = sine_train.compute_average_loss(model)
            print("Parameters:", degree, ", 1e", lr, " train loss:", train_loss)
            val_loss = sine_val.compute_average_loss(model)
            print("Parameters:", degree, ", 1e", lr, " val loss:", val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_degree = degree
                best_lr = lr

    print("Best Parameters:", degree, ',', lr)
    print("Best validation loss:", best_loss)

    best_model = PolynomialRegressionModel(best_degree, 10**best_lr)
    best_model.train(sine_train)

    util.RegressionDataset.plot_data(sine_train, best_model)


# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"

    def get_features(self, x):
        "*** YOUR CODE HERE ***"

    def get_weights(self):
        "*** YOUR CODE HERE ***"

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"


# PA4 Q4
def binary_classification():
    "*** YOUR CODE HERE ***"


# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"

    def get_features(self, x):
        "*** YOUR CODE HERE ***"

    def get_weights(self):
        "*** YOUR CODE HERE ***"

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"


# PA4 Q6
def multi_classification():
    "*** YOUR CODE HERE ***"


def main():
    linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()
