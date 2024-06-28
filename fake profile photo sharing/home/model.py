import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def ucb_algorithm(accept_count, reject_count):
    total_count = accept_count + reject_count
    ucb_score = pd.Series(np.zeros(len(total_count)), index=total_count.index)  
    
    
    for i in range(len(total_count)):
        if total_count.iloc[i] == 0:
            ucb_score.iloc[i] = 0.5
        else:
            ucb_score.iloc[i] = accept_count.iloc[i] / total_count.iloc[i] + np.sqrt(2 * np.log(total_count.iloc[i]) / (total_count.iloc[i]))
    
    return ucb_score


data = pd.read_csv('train.csv')


data['threshold'] = ucb_algorithm(data['accept_count'], data['reject_count'])


X = data[['group_size', 'accept_count', 'reject_count']]
y = data['threshold']


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'threshold1.pkl')


y_pred = model.predict(X)

# accuracy prediction score
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()

# Plotting residuals
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='green')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Plotting feature importance
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance, color='orange')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()