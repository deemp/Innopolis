import numpy as np
import pandas as pd


class Function:
    def method(self, *args):
        pass

class MyFunction(Function):
    @staticmethod
    def F(x):
        x0, y0 = MyFunction.x0, MyFunction.y0
        C = np.exp(-y0 / x0) - x0
        return -x * np.log(x + C)

    @staticmethod
    def f(x, y):
        return y / x - x * np.exp(y / x)

    def method(self, n, h):
        ys = np.empty(n)
        x0, F = MyFunction.x0, MyFunction.F
        for i in range(0, n):
            ys[i] = F(x0 + h * i)

        return ys

    # initial conditions
    x0, y0, X, N, n0, N0 = 1., 0., 1.5, 5, 10, 30

    @staticmethod
    def update(x0, y0):
        MyFunction.x0, MyFunction.y0 = x0, y0


class NumericalMethod(Function):
    def nxt(self, xi, yi, h):
        return yi

    def method(self, n, h):
        ys = np.full(n, MyFunction.y0)
        for i in range(0, n - 1):
            ys[i + 1] = self.nxt(MyFunction.x0 + h * i, ys[i], h)

        return ys

    def lte(self, n, h):
        lte = np.zeros(n)
        for i in range(0, n - 1):
            xi = MyFunction.x0 + h * i
            lte[i + 1] = np.abs(self.nxt(xi, MyFunction.F(xi), h) - MyFunction.F(xi + h))

        return lte


class Euler(NumericalMethod):
    def nxt(self, xi, yi, h):
        return yi + h * MyFunction.f(xi, yi)


class ImprovedEuler(NumericalMethod):
    def nxt(self, xi, yi, h):
        y = yi + h * MyFunction.f(xi, yi)
        return yi + h / 2 * (MyFunction.f(xi, yi) + MyFunction.f(xi + h, y))


class RungeKutta(NumericalMethod):
    def nxt(self, xi, yi, h):
        k1 = MyFunction.f(xi, yi)
        k2 = MyFunction.f(xi + h / 2, yi + h / 2 * k1)
        k3 = MyFunction.f(xi + h / 2, yi + h / 2 * k2)
        k4 = MyFunction.f(xi + h, yi + h * k3)
        return yi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Grid:
    # xs, Ys, EM, IEM, LTE, GTE
    @staticmethod
    def generate_data(x0, X, y0, N, n0, N0):

        # ============ putting definitions ============
        x0, X, y0 = float(x0), float(X), float(y0)
        MyFunction.update(x0, y0)

        n = N + 1
        h = (X - x0) / N

        exact_values = MyFunction().method

        em = Euler().method
        em_lte = Euler().lte

        iem = ImprovedEuler().method
        iem_lte = ImprovedEuler().lte

        rk = RungeKutta().method
        rk_lte = RungeKutta().lte

        # ============ collecting data for tab1 ============

        # ------------ calculating x-s ------------
        xs = np.empty(n)
        for i in range(n):
            xs[i] = x0 + i * h

        # ------------ calculating exact and approximate values ------------
        exact = exact_values(n, h)
        approx = np.array([em(n, h), iem(n, h), rk(n, h)])

        # ------------ calculating LTEs ------------
        lte = np.array([em_lte(n, h), iem_lte(n, h), rk_lte(n, h)])

        # ++++++++++++ gathering ++++++++++++
        tab1 = pd.DataFrame({'xs': xs, 'exact': exact,
                             'em_approx': approx[0], 'iem_approx': approx[1], 'rk_approx': approx[2],
                             'em_lte': lte[0], 'iem_lte': lte[1], 'rk_lte': lte[2]
                             })
        # print(tab1[['xs', 'iem_approx', 'iem_lte']])

        # ============ collecting data for tab2 ============

        # ------------ calculating n-s ------------
        ns = np.arange(n0, N0 + 1, 1)

        # ------------ calculating GTEs ------------
        n = N0 - n0 + 1
        gte = np.empty([3, n])
        for i in range(n):
            steps = n0 + i + 1
            h = (X - x0) / (steps - 1)
            exacts = exact_values(steps, h)
            gte[0][i] = np.max(np.abs(np.subtract(em(steps, h), exacts)))
            gte[1][i] = np.max(np.abs(np.subtract(iem(steps, h), exacts)))
            gte[2][i] = np.max(np.abs(np.subtract(rk(steps, h), exacts)))

        # ++++++++++++ gathering ++++++++++++
        tab2 = pd.DataFrame({'ns': ns, 'em_gte': gte[0], 'iem_gte': gte[1], 'rk_gte': gte[2]})
        # print(tab2)
        return (tab1, tab2)
