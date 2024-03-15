import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Diabetes.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,10].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 50)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

model.predict(X_test)


plt.scatter(X_train[:,0],y_train,color="yellow")
plt.plot(X_train[:,1],model.predict(X_train),color="blue")
plt.show()

error = model.score(X_train,y_train)
print(error)

