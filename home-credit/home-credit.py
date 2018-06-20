import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures


def clean_application_data(data, training=False):
    df = data
    le = LabelEncoder()

    # Label encode the columns that only have two values
    le.fit(df['NAME_CONTRACT_TYPE'])
    df['NAME_CONTRACT_TYPE'] = le.transform(df['NAME_CONTRACT_TYPE'])
    le.fit(df['FLAG_OWN_CAR'])
    df['FLAG_OWN_CAR'] = le.transform(df['FLAG_OWN_CAR'])
    le.fit(df['FLAG_OWN_REALTY'])
    df['FLAG_OWN_REALTY'] = le.transform(df['FLAG_OWN_REALTY'])

    # The rest either have missing values, or have more than two values
    df = pd.get_dummies(df)

    # Confusing that DAYS_BIRTH is negative, make it positive
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])

    return df


def add_poly_features(data):
    df = data

    # The ext source fields and the days birth fields are a good starting point
    # They have the highest p-value correlations with the target
    # Should try making polynomial features out of them


    return df


def fill_missing_values(data):
    df = data
    df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
    df['EXT_SOURCE_2'] = df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median())
    df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].fillna(df['DAYS_BIRTH'].median())
    return df


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


plt.interactive(False)

raw_application = pd.read_csv('application_train.csv')
train_application, test_application = split_train_test(raw_application, 0.2)

train_labels = train_application['TARGET']

# Looking at distribution of target classes
# print(raw_application['TARGET'].value_counts())
# clean_train_application['TARGET'].astype(int).plot.hist()
# Much more 0s than 1s (0 is repaid, 1 is didn't repay)

# Looking at missing feature values
# print(raw_application.isnull().sum().sort_values(ascending=False))

# Looking at types of columns
# print(raw_application.dtypes.value_counts())

# Number of unique classes in each categorical column
# print(raw_application.select_dtypes('object').apply(pd.Series.nunique, axis=0))

# Clean the data
train_application = clean_application_data(train_application, True)
test_application = clean_application_data(test_application)

# print(train_application.shape)
# print(test_application.shape)
# There are more features in the training data now because there were more values for some of the one-hot
# encodings in the training set. Need to align the training and test sets
train_application, test_application = train_application.align(test_application, join='inner', axis=1)
# print(train_application.shape)
# print(test_application.shape)

# Find correlations (p-values) between features and target
# correlations = train_application.corr()['TARGET'].sort_values()
# print(correlations)

# Fill missing values
train_application = fill_missing_values(train_application)

plt.show(block=True)