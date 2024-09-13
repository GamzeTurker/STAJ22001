import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("random forest regression dataset.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)


from sklearn.ensemble import RandomForestRegressor
random_forest=RandomForestRegressor(n_estimators=100,random_state=42)
random_forest.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu",random_forest.predict([[7.8]]) )

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=random_forest.predict(x_)

plt.scatter(x, y, color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()