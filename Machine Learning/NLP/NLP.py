import pandas as pd
import numpy as np
#%% import twitter data

data=pd.read_csv("gender_classifier.csv",encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender=[1 if each=="female" else 0 for each in data.gender]

#%% cleaning data
# regular expression RE mesela "[^a-zA-Z]"

import re
first_description=data.description[4]
description=re.sub("[^a-zA-Z]"," ",first_description)#a dan z ye ve A dan Z ye kadar olanları bul geri kalanları boşlukla  değiştirir
description=description.lower() #metni küçük harfe çevirir

#%%stopwords(irrelavent words) gereksiz kelimeler
import nltk #natural language tool kit
nltk.download("stopwords")#gereksiz kelimeleri indirdik (corpus diye bir klasöre indiriliyor)
from nltk.corpus import stopwords #nltk.corpustan gereksiz kelimeleri import ettik
#description=description.split()#boşluklara göre texti ayırır
description=nltk.word_tokenize(description)
#split kullanırsak "shouldn't " gibi kelimler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize kullanırsak ayrılır

#%%
#gereksiz kelimleri çıkar
description=[word for word in description if not word in set(stopwords.words("english"))]

#%%lemmatization ->kelimelerin köklerini bulma = loved->love
nltk.download('wordnet')
import nltk as nlp
lemma=nlp.WordNetLemmatizer()
description=[lemma.lemmatize(word) for word in description]
description=" ".join(description)#descriptiondaki kelimeleri  birleştirir

#%% data cleaning ->yukarıdaki işlemlerin hepsini data da yapıyoruz
nltk.download('punkt')
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma=nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)
    
#%% bag od words    
from sklearn.feature_extraction.text import CountVectorizer#bag of words oluştumak için kullandığımız metod
max_features=7500
count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix=count_vectorizer.fit_transform(description_list).toarray()
print("en sık kullanılan {} kelimeler :{}".format(max_features,count_vectorizer.get_feature_names_out()) )

#%%
y=data.iloc[:,0].values #male or female classes 
x=sparce_matrix
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

#%%naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

#%%predicition
y_pred=nb.predict(x_test)
print("accuary:",nb.score(y_pred.reshape(-1,1),y_test))

