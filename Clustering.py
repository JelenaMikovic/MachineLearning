import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

train_set_path = sys.argv[1]
test_set_path = sys.argv[2]

train_data = pd.read_csv(train_set_path)
test_data = pd.read_csv(test_set_path)

X_train = train_data.drop('region', axis=1)
X_test = test_data.drop('region', axis=1)

X_train = train_data.drop('Year', axis=1)
X_test = test_data.drop('Year', axis=1)

y_train = train_data['region']
y_test = test_data['region']

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

numeric_features = X_train.select_dtypes(include=np.number).columns
numeric_imputer = KNNImputer(n_neighbors=20)
#numeric_imputer = SimpleImputer(strategy='mean')

X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

gmm_params = {
    'n_components': 27,
    'covariance_type': 'diag',
    'init_params': 'random',
    'max_iter': 50,
    'n_init': 1,
    'random_state': 42,
    'reg_covar': 0.01,
    'tol': 0.01,
    'warm_start': False,
    'weights_init': None,
}

gmm = GaussianMixture(**gmm_params)
gmm.fit_predict(X_train)

y_predict = gmm.predict(X_test)

test_v_measure = v_measure_score(y_test, y_predict)

print(test_v_measure)