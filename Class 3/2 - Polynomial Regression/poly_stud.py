
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

dataset = pd.read_csv("students.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

degree = 4  
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

X_train_sorted = X_train[:, 0]  
y_train_sorted = y_train  

sorted_indices = np.argsort(X_train_sorted)
X_train_sorted = X_train_sorted[sorted_indices]
y_train_sorted = y_train_sorted[sorted_indices]

plt.scatter(X_train_sorted, y_train_sorted, color="red")
plt.plot(X_train_sorted, model.predict(X_train_poly[sorted_indices]), color="yellow")
plt.show()


rsquare_poly = r2_score(y_test, y_pred)
mse_poly = mean_squared_error(y_test, y_pred)

print("R-squared for Polynomial Regression:", rsquare_poly)
print("Mean Squared Error for Polynomial Regression:", mse_poly)
