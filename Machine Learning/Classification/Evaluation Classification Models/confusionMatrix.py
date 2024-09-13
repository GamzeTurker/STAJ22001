import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\Gamze\Desktop\Python ile Yapay Zeka\ML\KNN\data.csv")
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
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

print("score:",rf.score(x_text,y_text))

y_pred=rf.predict(x_text)
y_true=y_text

#%%confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true, y_pred)

#%%confusion matrix visualization
import seaborn as sns
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()