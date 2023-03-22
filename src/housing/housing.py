# ch2: Hands On Machine Learning
# resources from https://urldefense.com/v3/__https://www.kaggle.com/datasets/kathuman/housing__;!!NT4GcUJTZV9haA!sBuuYgQ5qnoRHkLW08WjrUjlZ0iRtm8cNgo6kUgHw70IbNqPhbmdzEy-kEMTj9O48MapFzg7trR5U9ph$

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# SK Learn design:
# Estimators - fit() method
# Transformers - transform() and fit_transform() methods
# Predictors - predict() method

"""
Class to add derived values to the data frame
"""
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAddr(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X.iloc[:, rooms_ix] / X.iloc[:, households_ix]
        population_per_household = X.iloc[:, population_ix] / X.iloc[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X.iloc[:, bedrooms_ix] / X.iloc[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def execute_regression(name, reg_algo, X_train, X_test, y_train, y_test):
    reg_algo.fit(X_train, y_train)
    reg_algo_predictions = pd.DataFrame(reg_algo.predict(X_test), columns=['y_predicted'])
    reg_algo_mae = mean_absolute_error(reg_algo_predictions, y_test)
    reg_algo_rmse = np.sqrt(mean_squared_error(reg_algo_predictions, y_test))
    reg_algo_scores = cross_val_score(reg_algo, X, y, scoring='neg_mean_squared_error', cv=10)
    reg_algo_rmse_scores = np.sqrt(-reg_algo_scores)
    display_scores(name, reg_algo_mae, reg_algo_rmse, reg_algo_rmse_scores)


def display_scores(name, mae, rmse, scores):
    print(' ')
    print('Model : ', name)
    print('MAE   : ', mae)
    print('RMSE  : ', rmse)
    print('Scores: ', scores)
    print('Mean  : ', scores.mean())
    print('Stdev : ', scores.std())



if __name__ == '__main__':
    raw_data = pd.read_csv('../../resources/housing.csv')
    raw_attributes = raw_data.columns
    print(raw_data['ocean_proximity'].value_counts())
    print(raw_data.info())
    print(raw_data.describe())
    print(raw_data['ocean_proximity'].value_counts())

    # Begin exploratory plotting
#    titles = np.array(['longitude', 'latitude',
#                       'housing_median_age', 'total_rooms',
#                       'total_bedrooms', 'population',
#                       'households', 'median_income',
#                       'median_house_value', 'ocean_proximity'])
#    colors = ['blue']
#    default_width = 4.8  # inches
#    plot_histograms('Kaggle Housing Data', titles, resources, 20, default_width, default_width * titles.size)
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

#    plt.scatter(X['median_income'], y, alpha=0.1)
#    plt.show()

    # End exploratory plotting


    # Begin data cleaning

    # Add in the new derived values

    # Pipeline setup
    cat_attributes = [ 'ocean_proximity']
    num_data = pd.DataFrame(raw_data.drop(columns = cat_attributes, axis=1))
    num_attributes = num_data.columns

    # Clean data manually.  Figure out the pipeline later.
    imputer = SimpleImputer(strategy='median')
    cleaned_data = pd.DataFrame(imputer.fit_transform(num_data), columns=num_attributes)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    encoded_data = pd.DataFrame(ohe.fit_transform(raw_data[cat_attributes]))
    cleaned_data = cleaned_data.join(encoded_data)
    addr = CombinedAttributesAddr(True)
    cleaned_data = pd.DataFrame(addr.fit_transform(cleaned_data))
    scalar = StandardScaler()
    cleaned_data = pd.DataFrame(scalar.fit_transform(cleaned_data))

    # Do a PCA for fun.  What is most important?
    # https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    # https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
    pca = PCA(n_components=16)
    pca.fit(cleaned_data)
    principal_data = pca.transform(cleaned_data)
    num_components = pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(num_components)]
    print([t for t in enumerate(pca.explained_variance_ratio_)])
    print([t for t in enumerate(pca.explained_variance_ratio_.cumsum())])
    # End data cleaning pipeline

    # Begin modeling pipeline
    # create test and train sets, taking median income stratification into account
    X = cleaned_data.copy()
    y = raw_data['median_house_value']
    y_actual = pd.DataFrame(y.values, columns=['y_actual'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    execute_regression('Generalized Boost Method', GradientBoostingRegressor(), X_train, X_test, y_train, y_test)
    execute_regression('Decision Tree', DecisionTreeRegressor(), X_train, X_test, y_train, y_test)
    execute_regression('XG Boost', XGBRegressor(), X_train, X_test, y_train, y_test)
    # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    execute_regression('Random Forest', RandomForestRegressor(n_estimators=50, random_state=1957), X_train, X_test, y_train, y_test)
    execute_regression('Linear Regression', LinearRegression(), X_train, X_test, y_train, y_test)

    # End modeling pipeline

