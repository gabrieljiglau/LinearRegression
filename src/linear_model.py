import numpy as np
from src.building_blocks import Dataset
from src.utils import calculate_slope, calculate_intercept, plot_regression_line, \
    calculate_linear_correlation, import_linear_dataset, calculate_error, calculate_gradient


class MultiVariateRegressor:

    def __init__(self, data: Dataset, learning_rate=0.03, epochs=25):

        if data is not None:
            self.data = data
        else:
            print('The provided dataset must not be empty')

        # w_0 + w_1*X_1 + w_2*X_2 + ... + w_n*X_n -> n + 1 parameters
        self.weights = [np.random.uniform(-1, 1) for _ in range((self.data.input_vars[0]) + 1)]
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self):
        """
        find the weight vector W=[w_0,w_1,…,w_n], that minimizes the mean squared error (MSE)
        between the predicted and actual target values
        :return:
        """

        x = self.data.input_vars
        y = self.data.output_vars

        for epoch in range(self.epochs):
            y_out = np.dot(x, self.weights)
            error = calculate_error(y_out, y)

            print(f"Current error {error}")

            for i in range(len(self.weights)):
                weight = self.weights[i]
                gradient = calculate_gradient(x, y, self.weights, i)
                weight -= self.learning_rate * gradient

        return self.weights


class LinearRegressor:

    def __init__(self, data: Dataset):

        if data is not None:
            self.data = data
        else:
            print('The provided dataset must not be empty')
        self.learned_alpha = 0
        self.learned_beta = 0

    def fit(self):

        """
        fit the best line y = alpha + x * beta, where alpha - the intercept and beta - the slope
        :return: the coefficients and the best fit line
        """

        if not self.data.is_linear_dataset_consistent():
            print("There aren't as many x-es as y-s or vice-versa in the dataset")
            return

        x_in = self.data.input_vars
        y_out = self.data.output_vars

        num_observations = len(x_in)

        x_squared = [x * x for x in x_in]
        xy = [x_in[i] * y_out[i] for i in range(num_observations)]

        x_sum = np.sum(x_in)
        y_sum = np.sum(y_out)
        x_squared_sum = np.sum(x_squared)
        xy_sum = np.sum(xy)

        # the slope
        beta = calculate_slope(num_observations, xy_sum, x_sum, y_sum, x_squared_sum)

        # the intercept = the point where the graph function intersects the y-axis; these points satisfy x = 0
        alpha = calculate_intercept(num_observations, y_sum, beta, x_sum)

        self.learned_alpha = alpha
        self.learned_beta = beta
        return alpha, beta, alpha + beta * x_in

    def predict(self, new_x):
        return self.learned_alpha + self.learned_beta * new_x


if __name__ == '__main__':

    """
    height_and_weight = import_linear_dataset('height_weight.csv')
    model = LinearRegressor(height_and_weight)
    intercept, slope, y_pred = model.fit()

    pearson_coefficient = calculate_correlation(height_and_weight, beta=slope)

    print(f"intercept: {intercept} \nslope: {slope}\ny_pred: {y_pred}")

    plot_regression_line(height_and_weight, y_pred, "height(cm)", "weight(kg)")

    x_new = 163
    prediction = model.predict(x_new)
    print(f"For a person {x_new} cm tall, his weight is {prediction}")
    """

    temperature_dataset = import_linear_dataset('temperatures.csv')

    model = LinearRegressor(temperature_dataset)
    intercept, slope, y_pred = model.fit()

    pearson_coefficient = calculate_linear_correlation(temperature_dataset, beta=slope)

    print(f"intercept: {intercept} \nslope: {slope}\ny_pred: {y_pred}")

    plot_regression_line(temperature_dataset, y_pred, "observed days", "temperature(C)")

    x_new = 8
    prediction = model.predict(x_new)
    print(f"The weather prediction for the next day, at 12:00 is {prediction}")
