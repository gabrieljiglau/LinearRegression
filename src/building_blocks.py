import numpy as np
from typing import List

class Dataset:

    def __init__(self, input_vars: List[float], output_vars: List[float]):
        self._input_vars = np.array(input_vars)
        self._output_vars = np.array(output_vars)

    @property
    def input_vars(self):
        return self._input_vars

    @input_vars.setter
    def input_vars(self, new_vars):
        if new_vars is None:
            print("Empty input variables")
            return
        elif [element < 0 for element in new_vars]:
            print("Some input variables are negative")
            return
        self._input_vars = new_vars

    @property
    def output_vars(self):
        return self._output_vars

    @output_vars.setter
    def output_vars(self, new_vars):
        if new_vars is None:
            print("Empty input variables")
            return
        elif [element < 0 for element in new_vars]:
            print("Some output variables are negative")
            return
        self._output_vars = new_vars

    def is_linear_dataset_consistent(self):
        return True if len(self.input_vars) == len(self.output_vars) else False

    def is_multivariate_dataset_consistent(self):
        if self.input_vars.size == 0 or self.output_vars == 0:
            return False

        if self.input_vars.shape[0] != self.output_vars.shape[0]:
            return False

        # 2d array
        if len(self.input_vars.shape) != 2:
            return False

        return True
