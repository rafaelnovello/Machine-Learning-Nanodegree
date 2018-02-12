
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[2]:


train = pd.read_csv('data/train.csv')
train.head()


# In[3]:


train.info()


# In[4]:


bids = pd.read_csv('data/bids.csv')
bids.head()


# In[5]:


bids.info()


# In[6]:


train_bids = pd.merge(bids, train, how='left', on='bidder_id')
del train
del bids


# In[7]:


train_bids.head()


# In[8]:


train_bids.dropna(subset=['outcome'], inplace=True)


# In[9]:


train_bids.head()


# In[32]:


train_bids[['auction', 'merchandise', 'country', 'payment_account', 'address']].describe()


# In[36]:


train_bids.groupby(['outcome', 'merchandise'])['bid_id'].count().plot(kind='bar', figsize=(15,7))


# In[29]:


get_ipython().magic(u'pinfo train_bids.describe')

