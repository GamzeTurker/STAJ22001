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


#%%random foreat classification
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=1)
rfc.fit(x_train,y_train)
print("random forest algo result:",rfc.score(x_text,y_text))
