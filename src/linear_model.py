import numpy as np
from src.building_blocks import Dataset
from src.utils import calculate_slope, import_temperature_dataset, calculate_intercept


class LinearRegressor:

    def __init__(self, data: Dataset):

        if not data.is_dataset_consistent():
            print("There aren't as many x-es as y-s or vice-versa in the dataset")
            return

        self.data = data

    def fit(self):

        """
        fit the best line y = alpha + x * beta, where alpha is the intercept and beta is the slope
        :return:
        """

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

        # the intercept = the point where the graph function intersects the y-axis
        # these points satisfy x = 0
        alpha = calculate_intercept(num_observations, y_sum, beta, x_sum)

        return alpha + beta * x_in


if __name__ == '__main__':

    temperature_dataset = import_temperature_dataset()
    model = LinearRegressor(temperature_dataset)
    y = model.fit()
    print(y)
