
# coding: utf-8

# ### Feature Engineering To find the Top most relevant Features

# In[1]:

import pandas as pd
import numpy as np
import xgboost as xgb
from numpy import sort
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot 
get_ipython().magic(u'matplotlib inline')


# ### Loading the Required Datasets

# In[3]:

train_numeric = pd.read_csv("../data/train_numeric.csv", nrows=100000)

train_numeric = train_numeric.fillna(9999999)


# In[4]:

msk = np.random.rand(len(train_numeric)) < 0.7   # creating a 0.7, 0.3 training and test set 


train = train_numeric[msk]
test =train_numeric[~msk]

features = np.setdiff1d(list(train.columns), ['Response', 'Id'])

y = train.Response.ravel()
train = np.array(train[features])

y_test=test.Response.ravel()
test=np.array(test[features])

print('train: {0}'.format(train.shape))
prior = np.sum(y) / (1.*len(y))


# In[5]:

model = XGBClassifier()
model.fit(train, y)

#plot importance
plot_importance(model)
fig=pyplot.show()


# In[7]:

y_pred = model.predict(test)
predictions = [round(value) for value in y_pred]
accuracy = matthews_corrcoef(y_test, predictions)
print("matthews_corrcoef: %f" % (accuracy ))


top_indices=(model.feature_importances_).argsort()[::-1][:15]  # finding top 15 feature indices

print top_indices 
features_list = list(train_numeric.columns.values)
for i in range(len(top_indices)):
    print "Top "+str(i)+" Features is  "+features_list[top_indices[i]] # finding the corresponding feature names

thresholds = sort(model.feature_importances_)
thresholds =thresholds[::-1]

k=0
important_features=[]
while k <15:
    important_features.append(thresholds[k])
    k+=1

print ("15 Most Important Features")
print important_features
# Help sought from http://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
count = 0
for thresh in thresholds:
    count+=1
    if(count==30):
        break
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y)
    # eval model
    select_X_test = selection.transform(test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = matthews_corrcoef(y_test, predictions)
    print accuracy
    print("Thresh=%.3f, n=%d, matthews_corrcoef: %f" % (thresh, select_X_train.shape[1], accuracy))

