import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("linear_regression_dataset.csv",sep=";")
plt.scatter(df.deneyim, df.maas)
plt.show()


from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()#linear regression model
x=df.deneyim.values.reshape(-1,1)#numpya çevirip (14,) olan sütun ve sayısını sklearn anlamadığı için (14,1) e çevirdik
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)#model fit edilir
y_head=linear_reg.predict(x)

plt.plot(x,y_head,color="red")
plt.show()

#%%r_square
from sklearn.metrics import r2_score

print("r_square score:",r2_score(y,y_head))