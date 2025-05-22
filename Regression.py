#import package 
import pandas as pd 
diabetes_df = pd.read_csv("diabetes.csv")
print(diabetes_df.head())

X = diabetes_df.drop("Glucose", axis=1).values
y = diabetes_df["Glucose"].values
print(type(X), type(y))

X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)

X_bmi = X_bmi.reshape(-1,1)
print(X_bmi.shape)

import matplotlib.pyplot as plt 
plt.scatter(X_bmi,y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()


## use body mass index as independent varviable(x) and blood glucose as dependent variabe(y)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi,y)
prediction = reg.predict(X_bmi)
plt.scatter(X_bmi,y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
