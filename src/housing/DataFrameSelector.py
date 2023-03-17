from sklearn.base import BaseEstimator, TransformerMixin



"""
Class to handle Pandas DataFrame in SKLearn
"""
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, U, v=None):
        return self

    def transform(self, U, v=None):
        """
        Convert input Pandas DataFrame to a NumPy array
        :param U: Pandas DataFrame for independent variables
        :param v: Pandas DataFrame for dependent variables
        :return: NumPy arrays of independent and dependent variables
        """
        return (U[self.attribute_names].values, v)
