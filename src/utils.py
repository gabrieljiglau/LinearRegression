import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.building_blocks import Dataset


def calculate_gradient(x, y, weights, current_index):
    x_transpose = np.dot(np.transpose(x), -2)
    x_w = np.dot(x[current_index], weights[current_index])

    return np.dot(x_transpose, x_w)

def calculate_error(y_pred, y_true):

    err = []
    for i in range(y_pred.shape[0]):
        err.append(y_pred - y_true)
    return err

def plot_regression_line(dataset: Dataset, y_pred, x_name, y_name):

    x_in = dataset.input_vars
    y_out = dataset.output_vars

    # the points from the dataset
    plt.scatter(x_in, y_out, color='blue', label='Observed Data', marker='o')

    # regression line
    plt.plot(x_in, y_pred, color='red', label='Regression Line')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('Linear Regression: True vs Predicted')
    plt.legend()

    plt.grid(True)
    plt.show()

def import_linear_dataset(dataset_name) -> Dataset | None:

    """

    :param dataset_name: dataset in csv format with X being the first column, Y_target the second
    :return: the prepared for inference dataset or None if the format isn't proper
    """

    current_path = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(current_path, '..', 'datasets')
    csv_file = os.path.join(datasets_path, dataset_name)

    df = pd.read_csv(csv_file)

    if not len(df.columns) == 2:
        print("The dataset does not support linear regression")
        return None

    x = df[df.columns[0]]
    y = df[df.columns[1]]
    return Dataset(x, y)


def calculate_linear_correlation(dataset: Dataset, beta) -> float:
    """
    :param dataset: object of type Dataset
    :param beta: the slope of the calculated line
    :return: a floating point number in [0,1], calculated as ((beta * x_std)/y_std)^2
    """

    x = dataset.input_vars
    y = dataset.output_vars

    x_std = np.std(x)
    y_std = np.std(y)
    r = beta * x_std/y_std
    r2 = np.power(r, 2)

    if r2 > 0.65:
        print("There is a decent linear relationship between observed x and y")

    if r2 == 0:
        print("There is no linear relationship present")

    print(f"pearson_coefficient = {r2}")
    return r2

def calculate_slope(observations, xy_sum, x_sum, y_sum, x_squared_sum):

    numerator = observations * xy_sum - x_sum * y_sum
    denominator = observations * x_squared_sum - np.power(x_sum, 2)

    return numerator / denominator

def calculate_intercept(observations, y_sum, beta, x_sum):

    numerator = y_sum - beta * x_sum
    return numerator / observations
