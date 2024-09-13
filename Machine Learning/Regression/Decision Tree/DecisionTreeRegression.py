import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("decision+tree+regression+dataset.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor

tree_reg=DecisionTreeRegressor()  
tree_reg.fit(x,y)  #veriyi eğitiyoruz(ağaç modelini oluşturuyoruz)
tree_reg.predict([[6]])
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)#min(x) değerinden max(x) değerine kadar 0.01 aralıklarla x değeri oluştur
y_head=tree_reg.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribün level")
plt.ylabel("Ücret")
plt.show()
