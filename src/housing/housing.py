# ch2: Hands On Machine Learning
# resources from https://urldefense.com/v3/__https://www.kaggle.com/datasets/kathuman/housing__;!!NT4GcUJTZV9haA!sBuuYgQ5qnoRHkLW08WjrUjlZ0iRtm8cNgo6kUgHw70IbNqPhbmdzEy-kEMTj9O48MapFzg7trR5U9ph$

import matplotlib.pyplot as plt
import pandas as pd
from ch2_hands_on_ml import HousingCombinedAttributesAdder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# SK Learn design:
# Estimators - fit() method
# Transformers - transform() and fit_transform() methods
# Predictors - predict() method


if __name__ == '__main__':
    raw_data = pd.read_csv('../../resources/housing.csv')
    print(raw_data['ocean_proximity'].value_counts())
    X = raw_data.loc[:, raw_data.columns != 'median_house_value']
    y = raw_data.loc[:, 'median_house_value']
    print(X.info())
    print(X.describe())

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

    plt.scatter(X['median_income'], y, alpha=0.1)
    plt.show()

    # End exploratory plotting


    # Begin data cleaning pipeline
    # Pipeline setup
    num_attributes = list(X.drop('ocean_proximity', axis=1))
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('attributes_adder', HousingCombinedAttributesAdder()),
        ('standard_scalar', StandardScaler()),
    ])

    # https://datagy.io/sklearn-one-hot-encode/
    cat_attributes = ['ocean_proximity']
    cat_pipeline = Pipeline(steps=[
        ('one_hot_encoder', OneHotEncoder()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('cat_pipeline', cat_pipeline),
        ('num_pipeline', num_pipeline),
    ])

    X_cleaned = full_pipeline.fit_transform(X)
    # End resources cleaning

    # create test and train sets, taking median income stratification into account
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=1234)


