# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:55:44 2024

@author: Mohamed
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
data=pd.read_csv("C:/Users/Mohamed/Desktop/mahatech/Salaries.csv") 
data.head()
data.info()
###################################################################################
#process in data 
#drop col
data.drop({'Id','Notes','EmployeeName'},axis=1,inplace=True)
data.info()
###################################################################################
#Handling missing values 
data.isnull().sum()
#convert datatype
data['BasePay']=data['BasePay']!='Not Provided'
data['BasePay']=data['BasePay'].astype(float)
data['BasePay'].fillna(data['BasePay'].median,inplace=True)

data['OvertimePay']=data['OvertimePay']!='Not Provided'
data['OvertimePay']=data['OvertimePay'].astype(float)

data['OtherPay']=data['OtherPay']!='Not Provided'
data['OtherPay']=data['OtherPay'].astype(float)

data['Benefits']=data['Benefits']!='Not Provided'
data['Benefits']=data['Benefits'].astype(float)


data['Benefits'].fillna(data['Benefits'].median,inplace=True)
data.isnull().sum()

label_encoder = LabelEncoder()
data['Status']=label_encoder.fit_transform(data['Status'])
data["Status"].unique()

plt.scatter(data["TotalPayBenefits"], data["Status"])
plt.title("Correlation between TotalPayBenefits and Status")

data['Status'].fillna(data['Status'].median,inplace=True)
data.isnull().sum()
###################################################################################
#using label encoding
data['Agency']=label_encoder.fit_transform(data['Agency'])
data['JobTitle']=label_encoder.fit_transform(data['JobTitle'])
##################################################################################
#splitting data 
features=['JobTitle','OvertimePay','OtherPay','BasePay','Benefits','TotalPay','Year','Agency','Status']
x=data[features]
y=data['TotalPayBenefits']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

#importing algorithms
Al1 = LinearRegression()
Al2 = DecisionTreeRegressor(random_state=42)
Al3 = RandomForestRegressor(n_estimators=40, random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores_Al1 = cross_val_score(Al1, x_train, y_train, cv=kfold, scoring='r2')
scores_Al2 = cross_val_score(Al2, x_train, y_train, cv=kfold, scoring='r2')
scores_Al3 = cross_val_score(Al3, x_train, y_train, cv=kfold, scoring='r2')

print("Mean R2 Score for AL1 (Linear Regression):", np.mean(scores_Al1))
print("Mean R2 Score for AL2 (Decision Tree):", np.mean(scores_Al2))
print("Mean R2 Score for AL3 (Random Forest):", np.mean(scores_Al3))

# Fit and evaluate models on the test set
Al1.fit(x_train, y_train)
Al2.fit(x_train, y_train)
Al3.fit(x_train, y_train)

y_pred_Al1 = Al1.predict(x_test)
y_pred_Al2 = Al2.predict(x_test)
y_pred_Al3 = Al3.predict(x_test)

print("Test R2 Score for AL1 (Linear Regression):", r2_score(y_test, y_pred_Al1))
print("Test R2 Score for AL2 (Decision Tree):", r2_score(y_test, y_pred_Al2))
print("Test R2 Score for AL3 (Random Forest):", r2_score(y_test, y_pred_Al3))

print(y_pred_Al1)
print(y_pred_Al2)
print(y_pred_Al3)

y_test












