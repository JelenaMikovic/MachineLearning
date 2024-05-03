import sys
import numpy as np
import pandas as pd
import scipy as sp

def fit(X, Y, learning_rate = 0.1, iterations = 1000, l2_penalty = 1):
    rows, cols = X.shape
    weights = np.zeros(cols)
    bias = 0

    for i in range(iterations):
        y_pred = predict(X, weights, bias)

        #izvodi
        dWeights = ((-2 * (X.T).dot(Y - y_pred)) + (2 * l2_penalty * weights)) / rows
        dBias = -2 * np.sum(Y - y_pred) / rows

        weights = weights - learning_rate * dWeights
        bias = bias - learning_rate * dBias
    
    return weights, bias

def predict(X, weights, bias):
    return X.dot(weights) + bias

def calculate_rmse(y_true, y_predict):
    return np.sqrt(np.mean((y_true - y_predict) ** 2))

train_set_path = sys.argv[1]
test_set_path = sys.argv[2]

training_data = pd.read_csv(train_set_path, delimiter='\t')
test_data = pd.read_csv(test_set_path, delimiter='\t')

# encoding trening
training_data['Marka'] = pd.factorize(training_data['Marka'])[0]
training_data['Grad'] = pd.factorize(training_data['Grad'])[0]

# encoding test
test_data['Marka'] = pd.factorize(test_data['Marka'])[0]
test_data['Grad'] = pd.factorize(test_data['Grad'])[0]

# dodavanje columns koji fale
all_categories = {}
for column in ['Karoserija', 'Gorivo', 'Menjac']:
    all_categories[column] = set(training_data[column]).union(set(test_data[column]))

for column in all_categories:
    training_data[column] = pd.Categorical(training_data[column], categories=all_categories[column])
    test_data[column] = pd.Categorical(test_data[column], categories=all_categories[column])

training_data = pd.get_dummies(training_data, columns=['Karoserija', 'Gorivo', 'Menjac'])
test_data = pd.get_dummies(test_data, columns=['Karoserija', 'Gorivo', 'Menjac'])

# uklanjanje outliera
training_data = training_data[training_data['Cena'] <= 150000]

# matrica korelacije
#numeric_columns = training_data.select_dtypes(include=[np.number])
#correlation_matrix = numeric_columns.corr()
#print(correlation_matrix)
training_data.drop(columns=['Marka', 'Grad'], inplace=True)
test_data.drop(columns=['Marka', 'Grad'], inplace=True)

X_train = training_data.drop('Cena', axis=1) 
y_train = training_data['Cena']

X_test = test_data.drop('Cena', axis=1) 
y_test = test_data['Cena']

# standardizacija
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

y_train = np.log(y_train)
weights, bias = fit(X_train_scaled, y_train)

y_pred = predict(X_test_scaled, weights, bias)

rmse = calculate_rmse(y_test.values, np.exp(y_pred))
print(rmse)