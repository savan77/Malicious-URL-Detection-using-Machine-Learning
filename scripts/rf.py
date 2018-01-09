
# coding: utf-8

# # Import Dependencies

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import random
from sklearn.metrics import accuracy_score


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[4]:

#custom tokenizer for URLs. 
#first split - "/"
#second split - "-"
#third split - "."
#remove ".com" (also "http://", but we dont have "http://" in our dataset)
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

#function to remove "http://" from URL
def trim(url):
    return re.match(r'(?:\w*://)?(?:.*\.)?([a-zA-Z-1-9]*\.[a-zA-Z]{1,}).*', url).groups()[0]


# # Prepare Dataset

# In[5]:

#read from a file
data = pd.read_csv("data/dataNN.csv",',',error_bad_lines=False)	#reading file
data['url'].values


# In[6]:

#convert it into numpy array and shuffle the dataset
data = np.array(data)
random.shuffle(data)


# In[ ]:




# In[7]:

#convert text data into numerical data for machine learning models
y = [d[1] for d in data]
corpus = [d[0] for d in data]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
X = vectorizer.fit_transform(corpus)



# In[8]:

#split the data set inot train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train Machine Learning Models 

# In[9]:

#1 - Logistic Regression
#model = LogisticRegression()
#model.fit(X_train, y_train)


# In[10]:

#print(model.score(X_test,y_test))


# In[10]:

#save the model and vectorizer
#joblib.dump(model, "mal-logireg1.pkl", protocol=2)
#joblib.dump(vectorizer, "vectorizer1.pkl", protocol=2)


# In[16]:

#make prediction
#a = "http://www.savanvisalpara.com"
#aa = vectorizer.transform([trim(a)])
#s = model.predict(aa)
#s[0] #0 for good


# In[ ]:

#2 - SVM
#from sklearn import svm
#svcModel = svm.SVC()
#svcModel.fit(X_train, y_train)
#lsvcModel = svm.LinearSVC.fit(X_train, y_train)


# In[ ]:

#svcModel.score(X_test, y_test)


# In[11]:

from sklearn.ensemble import RandomForestClassifier


# In[ ]:

m = RandomForestClassifier(n_estimators=25)
m.fit(X_train, y_train)
print(m.score(X_test,y_test))




