import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



"""
Class to append recommended combined numerical attributes for data cleaning pipeline
"""
class HousingCombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, U, v=None):
        return self

    def transform(self, U, v=None):
        """
        Recommended additional numerical attributes
        :param U: Pandas DataFrame for independent variables
        :param v: Pandas DataFrame for dependent variables
        :return: Pandas data frame with recommended numerical attributes added
        """
        U['rooms_per_household'] = U['total_rooms'] / U['households']
        U['bedrooms_per_room'] = U['total_bedrooms'] / U['total_rooms']
        U['population_per_household'] = U['population'] / U['households']
        U['income_cat'] = np.ceil(U['median_income'] / 1.5)
        U['income_cat'].where(U['income_cat'] < 5, 5.0, inplace=True)
        U['id'] = U['longitude'] * 1000 + U['latitude']
        return (U, v)

