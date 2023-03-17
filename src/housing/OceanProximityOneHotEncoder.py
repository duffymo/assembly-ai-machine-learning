import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



"""
Class to implement one hot encoding for ocean proximity for data cleaning pipeline
"""
class OceanProximityOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, U, v=None):
        return self

    def transform(self, U, v=None):
        """
        create one-hot encoding for ocean proximity category
        see https://urldefense.com/v3/__https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python__;!!NT4GcUJTZV9haA!sBuuYgQ5qnoRHkLW08WjrUjlZ0iRtm8cNgo6kUgHw70IbNqPhbmdzEy-kEMTj9O48MapFzg7tqq1G-W3$
        :param U: Pandas DataFrame for independent variables
        :param v: Pandas DataFrame for dependent variables
        :return: Pandas data frame of independent variables with one hot encoded categorical variables and dependent variables.
        """
        frame = pd.DataFrame(U)
        return (pd.get_dummies(frame, columns=['ocean_proximity'], prefix='h2o_proximity'), v)

