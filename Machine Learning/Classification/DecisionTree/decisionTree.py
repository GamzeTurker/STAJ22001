import numpy as np
import pandas as pd

data=pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%normalization
x=(x_data-np.min(x_data)/(np.max(x_data)-np.min(x_data)))

#%%trasin test split
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,test_size=0.15,random_state=42)

#%%decision tree classification
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("score:",dt.score(x_text,y_text))