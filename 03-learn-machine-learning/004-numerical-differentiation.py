import numpy as np


def numerical_differentiation(f, x):
    h = 10e-4
    return (f(x + h) - f(x))/h


def second_numerical_differentiation(f, x):
    h = 10e-4
    return (f(x + h) - f(x-h)) / (2*h)


def f(x):
    return 2 * (x ** 2)


print(numerical_differentiation(f, 3))
print(second_numerical_differentiation(f, 3))
