import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from functools import partial
from matplotlib import cm


def f_x(x):
    return x ** 2


def c_n_integrand(n, x):
    return f_x(x) * np.cos(n * np.pi / 2 * x)


def d_n_integrand(n, x):
    return f_x(x) * np.sin(n * np.pi / 2 * x)


def fourier_one_term(x, y, n):
    c_n, _ = integrate.quad(partial(c_n_integrand, n), -1, 1)
    c_n /= np.sinh(n * np.pi)
    d_n, _ = integrate.quad(partial(d_n_integrand, n), -1, 1)
    d_n /= np.sinh(n * np.pi)
    return (
        (c_n * np.cos(n * np.pi * x / 2) + d_n * np.sin(n * np.pi * x /  2)) *
        np.sinh(n * np.pi / 2 * (1 - y))
    )


def u_x_y(x, y):
    return sum(fourier_one_term(x, y, n) for n in range(1, 100))


def grid():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)
    u = u_x_y(x, y)
    surf = ax.plot_surface(x, y, u, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    ax.zaxis.set_major_formatter("{x:.02f}")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


grid()
