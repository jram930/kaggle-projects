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


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.RandomState(seed=42).permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def clean_data(data):
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

    return df


# Read in the data
raw = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

# Split into training and cross validation sets
train, cv = split_train_test(raw, 0.2)

# Clean the data
clean_all = clean_data(raw)
clean_train = clean_data(train)
clean_cv = clean_data(cv)
clean_test = clean_data(raw_test)

# Pull out the labels. This is the Survived column
y_all = clean_all['Survived']
y_train = clean_train['Survived']
y_cv = clean_cv['Survived']

# Pull out the features. This is every column except for Survived
X_all = clean_all.drop(['Survived'], axis=1)
X_train = clean_train.drop(['Survived'], axis=1)
X_cv = clean_cv.drop(['Survived'], axis=1)

# Data is prepped for training, train the model
tree_model = DecisionTreeClassifier()
tree_grid_model = GridSearchCV(tree_model, [{'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [5,10,15,20,None], 'max_features': [2,4,6,7]}])
tree_ada_model = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=7, splitter='random'))
forest_model = RandomForestClassifier()
forest_grid_model = GridSearchCV(forest_model, [{'bootstrap': [True, False], 'n_estimators': [2,4,6,8,10,12], 'criterion': ['gini', 'entropy'], 'max_depth': [5,10,15,20,None], 'max_features': [2,4,6,7]}])
forest_ada_model = AdaBoostClassifier(RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=5, max_features=2, n_estimators=8))
voting_model = VotingClassifier(estimators=[('forest', forest_model), ('forest_grid', forest_grid_model), ('forest_ada', forest_ada_model), ('tree', tree_model), ('tree_grid', tree_grid_model), ('tree_ada', tree_ada_model)], voting='hard')

best_score = -1
best_model = tree_model
for model in (tree_model, tree_grid_model, tree_ada_model, forest_model, forest_grid_model, forest_ada_model, voting_model):
    model.fit(X_all, y_all)
    y_pred = model.predict(X_all)
    accuracy = accuracy_score(y_all, y_pred)
    print(model.__class__.__name__, accuracy)
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

print(tree_grid_model.best_params_)
print(forest_grid_model.best_params_)

print('Best model is ', best_model.__class__.__name__, ' with accuracy of ', best_score)

# Run on test set and output to csv file
pred_test_y = forest_grid_model.predict(clean_test)
result = clean_test.reset_index()
result['Survived'] = pred_test_y
result = result[['PassengerId', 'Survived']]
result = result.set_index('PassengerId')
result.to_csv('result.csv')

