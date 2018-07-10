import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


yelp=pd.read_csv('yelp.csv')


# In[14]:


yelp.head()


# In[15]:


yelp.info()


# In[16]:


yelp.describe()


# In[17]:


yelp['text length']=yelp['text'].apply(len)


# In[18]:


yelp.head()


# In[19]:


#lenght
yelp['text length'].describe() #max leangth is 4997.00


# **Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**

# In[27]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')


# In[28]:


sns.boxplot(data=yelp,x='stars',y='text length')


# In[29]:


sns.countplot(x='stars',data=yelp)


# In[34]:


st=yelp.groupby('stars').mean()
st


# In[45]:


st.corr()


# In[46]:


sns.heatmap(st.corr(),cmap='viridis',annot=True)


# ## NLP Classification 
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

# In[47]:


yelp_class=yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[48]:


X=yelp_class['text']
y=yelp_class['stars']


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[51]:


X=cv.fit_transform(X)


# ## Train Test Split
# 

# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# ## Training a Model
# 
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

# In[59]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[60]:


nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 

# In[61]:


pred=nb.predict(X_test)


# In[63]:


from sklearn.metrics import classification_report,confusion_matrix


# In[65]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[1]:


#By TF-IDF


# # Using Text Processing
# 

# In[66]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[70]:


from sklearn.pipeline import Pipeline


# In[99]:


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# ## Using the Pipeline
# 

# In[100]:


X = yelp_class['text']
y = yelp_class['stars']


# In[101]:



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[102]:


pipeline.fit(X_train,y_train)


# ### Predictions and Evaluation
# 

# In[103]:


pred=pipeline.predict(X_test)


# In[105]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


