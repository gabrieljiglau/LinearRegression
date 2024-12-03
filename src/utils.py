import numpy as np
from src.building_blocks import Dataset

def import_temperature_dataset() -> Dataset:

    x_in = list(i for i in range(6))
    y_out = [6, 5, 3, 4, 3, 1]

    return Dataset(x_in, y_out)

def calculate_correlation(x, y, beta) -> float:
    """
    :param x: InputVariables
    :param y: OutputVariables
    :param beta: the slope of the calculated line
    :return: a floating point number in [0,1], calculated as ((beta * x_std)/y_std)^2
    """

    x_std = np.std(x)
    y_std = np.std(y)
    r = beta * x_std/y_std
    r2 = np.power(r, 2)

    if r2 > 0.65:
        print("There is a decent linear relationship between observed x and y")

    if r2 == 0:
        print("There is no linear relationship present")

    return r2

def calculate_slope(observations, xy_sum, x_sum, y_sum, x_squared_sum):

    numerator = observations * xy_sum - x_sum * y_sum
    denominator = observations * x_squared_sum - np.power(x_sum, 2)

    return numerator / denominator

def calculate_intercept(observations, y_sum, beta, x_sum):

    numerator = y_sum - beta * x_sum
    return numerator / observations

