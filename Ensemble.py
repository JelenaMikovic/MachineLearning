import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_set_path = sys.argv[1]
test_set_path = sys.argv[2]

train_data = pd.read_csv(train_set_path)
test_data = pd.read_csv(test_set_path)

X_train = train_data.drop('Genre', axis=1)
X_test = test_data.drop('Genre', axis=1)

y_train = train_data['Genre']
y_test = test_data['Genre']

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.0001, random_state = 42)

numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(exclude=np.number).columns

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])

X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

pca = PCA()
X_train = pca.fit_transform(X_train.toarray())
X_test = pca.transform(X_test.toarray())

random_forest = RandomForestClassifier(min_samples_leaf = 2, n_estimators = 200, min_samples_split = 10, random_state = 42)
ada_boost = AdaBoostClassifier(learning_rate = 1, n_estimators = 200, random_state = 42)
logistic_regression = LogisticRegression(max_iter = 500, random_state = 42)

voting_classifier = VotingClassifier(estimators = [('rf', random_forest), ('adaboost', ada_boost), ('lr', logistic_regression)], voting = 'soft', weights = [2, 1, 1])

voting_classifier.fit(X_train, y_train)

y_pred = voting_classifier.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average = 'macro')

print(macro_f1)