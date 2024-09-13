import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
data=pd.read_csv("C:/Users/Gamze/Desktop/Python ile Yapay Zeka/ML/KNN/data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
#%%
M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]
#%%
plt.scatter(M.radius_mean,M.area_mean,color="red",label="kotu")
plt.scatter(B.radius_mean,B.area_mean,color="green",label="iyi")
plt.legend()
plt.show()
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#%%
data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)
#%%
#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,test_size=0.3,random_state=1)
#%%
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
#%%tezt
print("print accuary of naive bayes algo:",nb.score(x_text,y_text))
