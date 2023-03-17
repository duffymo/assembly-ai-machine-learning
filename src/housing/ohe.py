import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79

if __name__ == '__main__':
    df = pd.read_csv('../../resources/StudentsPerformance.csv')
    print(df.info())
    print(df.describe())

    # five categorical independent variables, three numeric dependent variables
    # calculate correlation for math, reading, and writing scores
    numerical_attributes = ['math score', 'reading score', 'writing score']
    numerical_data = df[numerical_attributes]
    categorical_attributes = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']
    categorical_data = df[categorical_attributes]
    ordinal_attributes = ['parental level of education']
    ordinal_data = df[ordinal_attributes]

    corr_matrix = numerical_data.corr()
    # https://stackoverflow.com/questions/31698861/add-column-to-the-end-of-pandas-dataframe-containing-average-of-previous-data
    df = df.assign(mean_score = df[numerical_attributes].mean(axis=1, numeric_only=True))

    # Begin data analysis
#    score_by_gender = numeric_data_with_mean_score.groupby('gender')['mean_score'].mean()
#    score_by_race   = numeric_data_with_mean_score.groupby('race/ethnicity')['mean_score'].mean()
#    score_by_edu    = numeric_data_with_mean_score.groupby('parental level of education')['mean_score'].mean()
#    score_by_lunch  = numeric_data_with_mean_score.groupby('lunch')['mean_score'].mean()
#    score_by_prep   = numeric_data_with_mean_score.groupby('test preparation course')['mean_score'].mean()
#
#    c = ['red', 'orange', 'yellow', 'green', 'blue']
#    for i in range(0, len(categorical_attributes)):
#        plt.bar(numeric_data_with_mean_score[categorical_attributes[i]], numeric_data_with_mean_score['mean_score'])
#        plt.show()
#
#    # https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
#    plt.hist(numeric_data_with_mean_score['mean_score'], bins=50, color='red')
#    # https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
#    sns.displot(numeric_data_with_mean_score['mean_score'], color='dodgerblue')
#    plt.show()
    # End exploratory data analysis

    # Begin data cleaning
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).set_output(transform='pandas')

    encoded_data = pd.DataFrame(ohe.fit_transform(df[categorical_attributes]))
    df = df.join(encoded_data)
    encoded_data = pd.DataFrame(ohe.fit_transform(df[ordinal_attributes]))
    df = df.join(encoded_data)
    # End data cleaning

