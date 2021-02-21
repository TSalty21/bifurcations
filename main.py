import numpy as np
import matplotlib
matplotlib.use('TKAgg')

from matplotlib import pyplot as plt

from core.map_base import MapBase


class Logistic(MapBase):
    def __init__(self, r, x0, **kwargs):
        super(Logistic, self).__init__(**kwargs)

        self.name = "Logistic"

        self.params = {
            "r": r
        }

        self.vars = {
            "x_n": x0
        }

    def iterate(self, x_n=None):
        if x_n is not None:
            return self.params["r"] * x_n * (1.0 - x_n),
        else:
            raise ValueError("Iteration variable can't be None!")


class Gauss(MapBase):
    def __init__(self, alpha, beta, x0, **kwargs):
        super(Gauss, self).__init__(**kwargs)

        self.name = "Gauss"

        alpha, beta = np.meshgrid(alpha, beta)
        self.params = {
            "alpha": alpha,
            "beta": beta
        }

        self.vars = {
            "x_n": x0
        }

    def iterate(self, x_n=None):
        if x_n is not None:
            return np.exp(-self.params["alpha"] * np.square(x_n)) + self.params["beta"],
        else:
            raise ValueError("Iteration variable can't be None!")


class Henon(MapBase):
    def __init__(self, a, b, x0, y0, **kwargs):
        super(Henon, self).__init__(**kwargs)

        self.name = "Henon"

        a, b = np.meshgrid(a, b)
        self.params = {
            "a": a,
            "b": b
        }

        self.vars = {
            "x_n": x0,
            "y_n": y0
        }

    def iterate(self, x_n=None, y_n=None):
        if x_n is not None and y_n is not None:
            return y_n + 1 - self.params["a"] * np.square(x_n), self.params["b"] * x_n
        else:
            raise ValueError("Iteration variables can't be None!")


class Duffing(MapBase):
    def __init__(self, a, b, x0, y0, **kwargs):
        super(Duffing, self).__init__(**kwargs)

        self.name = "Duffing"

        a, b = np.meshgrid(a, b)
        self.params = {
            "a": a,
            "b": b
        }

        self.vars = {
            "x_n": x0,
            "y_n": y0
        }

    def iterate(self, x_n=None, y_n=None):
        if x_n is not None and y_n is not None:
            return y_n, -self.params["b"] * x_n + self.params["a"] * y_n - np.square(y_n)
        else:
            raise ValueError("Iteration variables can't be None!")


if __name__ == "__main__":
    print("*** Bifurcation diagrams ***")

    n = 1000
    iterations = 1000

    # r = np.linspace(2.7, 4, n)
    # x0 = 1e-5 * np.ones([n])
    # logistic = Logistic(r, x0, last_num=200)
    # logistic.plot_bifurcation_diagram(iterations)

    alpha = np.linspace(3, 12, n)
    beta = np.linspace(-1, 1, n)
    x0 = 1e-5 * np.ones([n, n])
    gauss = Gauss(alpha, beta, x0)
    gauss.plot_bifurcation_diagram(iterations)
    #
    # a = np.linspace(1, 1.4, n)
    # b = np.linspace(0.29, 0.31, n)
    # x0 = 1e-5 * np.ones([n, n])
    # y0 = 1e-5 * np.ones([n, n])
    # henon = Henon(a, b, x0, y0)
    # henon.plot_bifurcation_diagram(iterations)
    #
    # a = np.linspace(2.5, 3.8, n)
    # b = np.linspace(0, 0.5, n)
    # x0 = 1e-5 * np.ones([n, n])
    # y0 = 1e-5 * np.ones([n, n])
    # duffing = Duffing(a, b, x0, y0)
    # duffing.plot_bifurcation_diagram(iterations)

    plt.show()
