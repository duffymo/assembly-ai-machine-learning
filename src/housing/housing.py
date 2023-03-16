# ch2: Hands On Machine Learning
# data from https://www.kaggle.com/datasets/kathuman/housing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class HousingCombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Suggestions for new variables.  This is the class you should put them in.
        X['rooms_per_household'] = encoded_data['total_rooms'] / encoded_data['households']
        X['bedrooms_per_room'] = encoded_data['total_bedrooms'] / encoded_data['total_rooms']
        X['population_per_household'] = encoded_data['population'] / encoded_data['households']
        X['income_cat'] = np.ceil(encoded_data['median_income'] / 1.5)
        X['income_cat'].where(encoded_data['income_cat'] < 5, 5.0, inplace=True)
        X['id'] = encoded_data['longitude'] * 1000 + encoded_data['latitude']
        return X

if __name__ == '__main__':
    raw_data = pd.read_csv('../../resources/housing.csv')
    print(raw_data['ocean_proximity'].value_counts())
    # create one-hot encoding for ocean proximity category
    # https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    encoded_data = pd.get_dummies(raw_data, columns=['ocean_proximity'], prefix='h2o_proximity')
    attributes_adder = HousingCombinedAttributesAdder()
    encoded_data = attributes_adder.transform(encoded_data)

    X = encoded_data.loc[:, encoded_data.columns != 'median_house_value']
    y = encoded_data.loc[:, 'median_house_value']

    # Begin data cleaning
    # Replace NA values with median using SKLearn Imputer
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X)
    for i in range(len(imputer.feature_names_in_)):
        print(imputer.feature_names_in_[i], ' median: ', imputer.statistics_[i])
    X = pd.DataFrame(imputer.transform(X), columns=X.columns)
    # End data cleaning

    print(X.info())
    print(X.describe())

    # create test and train sets, taking median income stratification into account
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Begin exploratory plotting

#    titles = np.array(['longitude', 'latitude',
#                       'housing_median_age', 'total_rooms',
#                       'total_bedrooms', 'population',
#                       'households', 'median_income',
#                       'median_house_value', 'ocean_proximity'])
#    colors = ['blue']
#    default_width = 4.8  # inches
#    plot_histograms('Kaggle Housing Data', titles, data, 20, default_width, default_width * titles.size)
#

#    X_train.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#                 s=X_train['population']/100, label='population',
#                 c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
#    plt.legend()
#    plt.show()

#    corr_matrix = X_train.corr()
#    print(corr_matrix['median_house_value'].sort_values(ascending=False))

#    scatter_attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
#    scatter_matrix(encoded_data[scatter_attributes], figsize=(12, 8), alpha=0.2)
#    plt.show()

    encoded_data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    plt.show()

    # End exploratory plotting


