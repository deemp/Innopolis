import numpy as np
import pandas as pd


class Function:
    # initial conditions
    x0, X, y0, N, n0, N0 = 1., 8., 0.0, 10, 10, 30

    @staticmethod
    def update(x0, y0):
        Function.x0, Function.y0 = x0, y0

    def values(self, n, h):
        ys = np.full(n, self.y0)
        for i in range(0, n - 1):
            ys[i + 1] = self.nxt(self.x0 + h * i, ys[i], h)

        return ys

    def f(self, x, y):
        return y / x - x * np.exp(y / x)

    def nxt(self, xi, yi, h):
        return yi


class ExactSolution(Function):
    @staticmethod
    def F(x):
        x0, y0 = Function.x0, Function.y0
        C = np.exp(-y0 / x0) - x0
        return -x * np.log(x + C)

    def nxt(self, xi, yi, h):
        return self.F(xi + h)


class NumericalMethod(Function):

    def ltes(self, n, h):
        ltes = np.zeros(n)
        for i in range(0, n - 1):
            xi = self.x0 + h * i
            ltes[i + 1] = np.abs(self.nxt(xi, ExactSolution.F(xi), h) - ExactSolution.F(xi + h))

        return ltes


class Euler(NumericalMethod):
    def nxt(self, xi, yi, h):
        return yi + h * self.f(xi, yi)


class ImprovedEuler(NumericalMethod):
    def nxt(self, xi, yi, h):
        y = yi + h * self.f(xi, yi)
        return yi + h / 2 * (self.f(xi, yi) + self.f(xi + h, y))


class RungeKutta(NumericalMethod):
    def nxt(self, xi, yi, h):
        k1 = self.f(xi, yi)
        k2 = self.f(xi + h / 2, yi + h / 2 * k1)
        k3 = self.f(xi + h / 2, yi + h / 2 * k2)
        k4 = self.f(xi + h, yi + h * k3)
        return yi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Model:
    @staticmethod
    def get_state(x0, X, y0, N, n0, N0):

        # putting definitions and updating Model
        Function.update(x0, y0)

        exact = ExactSolution().values
        methods = (Euler().values, ImprovedEuler().values, RungeKutta().values)
        ltes = (Euler().ltes, ImprovedEuler().ltes, RungeKutta().ltes)

        # gathering plot data for tabs
        tab1 = Model.tab1_data(x0, X, N, exact, methods, ltes)
        tab2 = Model.tab2_data(x0, X, n0, N0, exact, methods)

        return tab1, tab2

    @staticmethod
    def tab1_data(x0, X, N, exact, methods, ltes):
        # collecting data for tab1

        n = N + 1
        h = (X - x0) / N

        # calculating x-s
        xs = np.empty(n)
        for i in range(n):
            xs[i] = x0 + i * h

        # calculating exact, approximate values, and LTE-s
        exacts = exact(n, h)
        approx = np.array([_(n, h) for _ in methods])
        lte = np.array([_(n, h) for _ in ltes])

        # gathering
        tab1 = pd.DataFrame({'xs': xs, 'exact': exacts,
                             'em_approx': approx[0], 'iem_approx': approx[1], 'rk_approx': approx[2],
                             'em_lte': lte[0], 'iem_lte': lte[1], 'rk_lte': lte[2]
                             })
        # view tab1 data:
        # print(tab1[['xs', 'exact', 'em_approx', 'em_lte']])
        return tab1

    @staticmethod
    def tab2_data(x0, X, n0, N0, exact, methods):
        # collecting data for tab2

        # calculating n-s
        ns = np.arange(n0, N0 + 1, 1)

        # calculating GTEs
        n = N0 - n0 + 1

        gte = np.empty([3, n])
        for i in range(n):
            steps = n0 + i + 1
            h = (X - x0) / (steps - 1)
            for j, method in zip([0, 1, 2], methods):
                gte[j][i] = np.max(np.abs(np.subtract(method(steps, h), exact(steps, h))))

        # gathering
        tab2 = pd.DataFrame({'ns': ns, 'em_gte': gte[0], 'iem_gte': gte[1], 'rk_gte': gte[2]})

        # view tab2 data:
        # print(tab2[['ns', 'em_gte']])

        return tab2
