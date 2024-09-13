import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/Gamze/Desktop/Python ile Yapay Zeka/ML/Regression/RandomForestRegression/random forest regression dataset.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)


from sklearn.ensemble import RandomForestRegressor
random_forest=RandomForestRegressor(n_estimators=100,random_state=42)
random_forest.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu",random_forest.predict([[7.8]]) )

y_head=random_forest.predict(x)

#r_square
from sklearn.metrics import r2_score

print("r_score:",r2_score(y,y_head))
