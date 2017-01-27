
# coding: utf-8

# In[53]:

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score


# In[54]:

df=pd.read_csv("MovieReview_cells_labelled.txt", sep='\t', names=['txt', 'liked'])


# In[55]:

df.head()


# In[56]:

#TFIDF Vectorizer, just like before
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[57]:

#in this case our dependent variable will be liked as 0 (didn't like the movie) or  1 (like the movie)
y=df.liked


# In[58]:

#convert df.txt from text to features
X=vectorizer.fit_transform(df.txt)


# In[59]:

#3000 observations x ......
print y.shape
print X.shape


# In[60]:

#test train_split as usual 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[61]:

#we will train a naive_bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)


# In[62]:

#we can test our model's accuracy like this
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


# In[63]:

move_amazon_array=np.array(["its so cool to consider "])
move_amazon_vector = vectorizer.transform(move_amazon_array)
print clf.predict(move_amazon_vector)


# In[ ]:




# In[ ]:



