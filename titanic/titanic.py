import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


scaler = MinMaxScaler()

def clean_data(data, training=False):
    # Set the index to the passenger ID
    df = data.set_index('PassengerId')

    # Drop the two rows that are missing embarked data
    df = df.dropna(subset=['Embarked'])

    # Drop the cabin column because it is missing so much data
    df = df.drop(['Cabin'], axis=1)

    # Replace the missing ages with the mean age
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # Replace the missing fares with the mean fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # Drop the name field. Doesn't seem important
    df = df.drop(['Name'], axis=1)

    # Replace male with 1 and female with 0
    df['Sex'] = df['Sex'].replace('male', 1)
    df['Sex'] = df['Sex'].replace('female', 0)

    # Drop the ticket column for now because it looks meaningless
    df = df.drop(['Ticket'], axis=1)

    # Replace S with 0, C with 1, and Q with 2 for the Embarked field
    df['Embarked'] = df['Embarked'].replace('S', 0)
    df['Embarked'] = df['Embarked'].replace('C', 1)
    df['Embarked'] = df['Embarked'].replace('Q', 2)

    if training:
        scaler.fit(df[['Age', 'SibSp', 'Parch', 'Fare']])

    df[['Age', 'SibSp', 'Parch', 'Fare']] = scaler.transform(df[['Age', 'SibSp', 'Parch', 'Fare']])

    return df


if __name__ == '__main__':

    # Read in the data
    raw = pd.read_csv('train.csv')
    raw_test = pd.read_csv('test.csv')

    # Split into training and cross validation sets
    train, cv = split_train_test(raw, 0.2)

    # Clean the data
    clean_train = clean_data(train, True)
    clean_val = clean_data(cv)
    clean_test = clean_data(raw_test)

    # Pull out the labels. This is the Survived column
    y_train = clean_train['Survived']
    y_val = clean_val['Survived']

    # Pull out the features. This is every column except for Survived
    X_train = clean_train.drop(['Survived'], axis=1)
    X_val = clean_val.drop(['Survived'], axis=1)

    # Data is prepped for training, train the model
    print('Training model...')
    model = GridSearchCV(RandomForestClassifier(), [{'bootstrap': [True, False], 'n_estimators': [8,10,15,20,50], 'criterion': ['gini', 'entropy'], 'max_depth': [5,10,15,20,None], 'max_features': [4,6,7]}], cv=5, verbose=1, n_jobs=8)
    # model = GridSearchCV(SVC(), [{'C': [0.1,0.5,1.0,2.0,5.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2,3,4,5]}], verbose=1, n_jobs=8, cv=5)
    model.fit(X_train, y_train)
    print('Done training!')

    pred_val_y = model.predict(X_val)
    print('Accuracy: ', accuracy_score(y_val, pred_val_y))

    # Run on test set and output to csv file
    pred_test_y = model.predict(clean_test)
    result = clean_test.reset_index()
    result['Survived'] = pred_test_y
    result = result[['PassengerId', 'Survived']]
    result = result.set_index('PassengerId')
    result.to_csv('result.csv')

