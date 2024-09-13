from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris=load_iris()
x=iris.data
y=iris.target

#%% normalization
x=(x-np.min(x)/(np.max(x)-np.min(x)))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


#%% K fold cross validation  k=10
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=knn, X=x_train,y=y_train,cv=10)
print("average accuracy:",np.mean(accuracies))
print("average std:",np.std(accuracies))#datanın ne kadar tutatlı olduğu

#%%
knn.fit(x_train,y_train) 
print("test accuracy:",knn.score(x_test,y_test))
#%% Grid search cross validation
from sklearn.model_selection import GridSearchCV
grid={"n_neighbors":np.arange(1, 50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)
#%% print hyperparameter KNN algoritmasındaki K değeri
print("tuned hyperparameter K: ",knn_cv.best_params_)#ayarlanarak bulunan parametre
print("tuned parametreye göre en iyi accuracy (best score):",knn_cv.best_score_)
#%% grid search CV with logistic regression
x=x[:100,:]
y=y[:100]
#%% normalization
x=(x-np.min(x)/(np.max(x)-np.min(x)))
#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#%% grid search CV with logistic regression
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3, 3,7),"penalty":["l1","l2"]}
lr=LogisticRegression()
lr_cv=GridSearchCV(lr, grid,cv=10)
lr_cv.fit(x_train,y_train)
print("tuned hyperparameters(best parameters):",lr_cv.best_params_)
print("accuracy:",lr_cv.best_score_)
#%%
lr2=LogisticRegression(C=0.01,penalty="l2")
lr2.fit(x_train,y_train)
print("score:",lr2.score(x_test, y_test))