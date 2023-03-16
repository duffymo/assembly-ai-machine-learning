import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.DataFrame()
    df['values'] = np.linspace(0, 10, 11)
    print('Before standardizing: ', df)
    print('Sum                 : ', df['values'].sum())
    print('# values            : ', df['values'].shape)
    print('Mean                : ', df['values'].mean())
    print('Variance            : ', df['values'].var())
    print('Sqrt(Variance)      : ', np.sqrt(df['values'].var()))
    print('Standard deviation  : ', df['values'].std())
    scaler = StandardScaler ()
    df['standardized'] = scaler.fit_transform(df ['values'].array.reshape (-1, 1))
    print('After  standardizing: ', df)
    df['z-score'] = zscore(df['values'])
    print('Z Score             : ', df['z-score'])
    