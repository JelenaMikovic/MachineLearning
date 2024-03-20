import sys
import numpy as np
import pandas as pd

def calculate_rmse(y_true, y_predict):
    return np.sqrt(np.mean((y_true - y_predict) ** 2))

def remove_outliers(df, columns, threshold=1.0):
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    return df[(z_scores < threshold).all(axis=1)]

def batch_gradient_descent(x, y, learning_rate=0.5, epochs=40):
    m, n = x.shape
    slope = np.ones(n) 
    costs = [] 

    for _ in range(epochs):
        pred = np.matmul(x, slope)
        J = 1 / 2 * np.mean((pred - y) ** 2)
        costs.append(J) 

        for i in range(n):
            slope[i] -= learning_rate / m * np.sum((pred - y) * x[:, i])

    return costs, slope

def fit(x, y):
    x_log = np.log(x)
    y_log = np.log(y)
    x_log = np.column_stack((np.ones(len(x_log)), x_log))
    cost, slope = batch_gradient_descent(x_log, y_log)
    return cost, slope

def predict(x, slope):
    x_log = np.log(x)
    x_log = np.column_stack((np.ones(len(x_log)), x_log)) 
    pred_log = np.matmul(x_log, slope)
    return np.exp(pred_log)

train_set_path = sys.argv[1]
test_set_path = sys.argv[2]

training_data = pd.read_csv(train_set_path)
test_data = pd.read_csv(test_set_path)

training_data = remove_outliers(training_data, ['X', 'Y'])

x_array = training_data['X'].values
y_array = training_data['Y'].values

cost, slope = fit(x_array, y_array)

y_pred = predict(test_data['X'].values, slope)

rmse = calculate_rmse(test_data['Y'], y_pred)
print(rmse)