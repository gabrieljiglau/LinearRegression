import numpy as np
from src.building_blocks import Dataset
from src.utils import calculate_slope, calculate_intercept, plot_regression_line, \
    calculate_linear_correlation, import_dataset, calculate_error, calculate_gradient


class MultiVariateRegressor:

    def __init__(self, data: Dataset, learning_rate=0.5, epochs=3, batch_size=32):

        if data is not None:
            self.data = data
        else:
            print('The provided dataset must not be empty')

        num_features = self.data.input_vars.shape[1]

        # w_0 + w_1*X_1 + w_2*X_2 + ... + w_n*X_n -> n + 1 parameters
        self.weights = np.random.uniform(-1, 1, num_features + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self):
        """
        find the weight vector W=[w_0, w_1, ..., w_n] that minimizes the mean squared error (MSE)
        between the predicted and actual target values
        """
        x = self.data.input_vars
        y = self.data.output_vars
        x_augmented = np.c_[np.ones(x.shape[0]), x]  # Add 1s for the bias term (w_0)

        for epoch in range(self.epochs):
            for batch_start in range(0, len(x_augmented), self.batch_size):
                batch_x = x_augmented[batch_start: batch_start + self.batch_size]
                batch_y = y[batch_start: batch_start + self.batch_size]

                gradients = np.zeros(len(self.weights))
                for i in range(len(self.weights)):
                    gradients[i] = calculate_gradient(batch_x, batch_y, self.weights, i)

                self.weights -= self.learning_rate * gradients

            # after each epoch, calculate and print the error (MSE)
            y_out = np.dot(x_augmented, self.weights)
            error = calculate_error(y_out, y)
            print(f"Epoch {epoch + 1}, Current error: {error}")

        return self.weights

    def predict(self, x_in):

        x_in = np.array(x_in)

        if x_in.ndim == 1:
            x_in = x_in.reshape(1, -1)

        x_augmented = np.c_[np.ones(x_in.shape[0]), x_in]

        if len(self.weights) != x_augmented.shape[1]:
            raise ValueError(
                f"Weight vector has {len(self.weights)} elements, but input data has {x_augmented.shape[1]} features.")

        return np.dot(x_augmented, self.weights)


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

    house_dataset = import_dataset('house_prices.csv', False)
    model = MultiVariateRegressor(house_dataset)

    model.fit()

    new_house = [1268, 5, 2, 10]
    print(f"Predicted price for {new_house} is {model.predict(new_house)}")

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

    """
    temperature_dataset = import_dataset('temperatures.csv')

    model = LinearRegressor(temperature_dataset)
    intercept, slope, y_pred = model.fit()

    pearson_coefficient = calculate_linear_correlation(temperature_dataset, beta=slope)

    print(f"intercept: {intercept} \nslope: {slope}\ny_pred: {y_pred}")

    plot_regression_line(temperature_dataset, y_pred, "observed days", "temperature(C)")

    x_new = 8
    prediction = model.predict(x_new)
    print(f"The weather prediction for the next day, at 12:00 is {prediction}")
    """

