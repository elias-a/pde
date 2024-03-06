import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from more_itertools import unzip


def dalembert(f, g, x, t, c):
    f_term = 0.5 * (f(x - c * t) + f(x + c * t))
    integral, _ = integrate.quad(g, x - c * t, x + c * t)
    return f_term + 1 / 2 / c * integral


"""
Plot the d'Alembert formula for the solution to the wave equation.
The function f equals the initial displacement and g represents
the initial velocity. The function generates 6 plots at different
times. Optionally, specify the wave speed, c.
"""
def plot_dalembert(
        f,
        g,
        c=1,
        times=[0., 0.25, 0.5, 1., 1.5, 3.],
        x_min=-5,
        x_max=5):
    if c <= 0:
        raise Exception("The wave speed c must be a positive number.")
    if len(times) != 6:
        raise Exception("The array of times must have a length of 6.")
    fig, axes = plt.subplots(2, 3)
    for t, ax in zip(times, axes.flatten()):
        ax.plot(*[list(v) for v in unzip(
            [(x, dalembert(f, g, x, t, c))
             for x in np.linspace(x_min, x_max, 100)]
        )])
        ax.set_title(f"t = {t}")
    plt.show()
