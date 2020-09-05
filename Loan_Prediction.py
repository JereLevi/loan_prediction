#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[98]:


train=pd.read_csv("train.csv")


# In[99]:


train["Married"]=train["Married"].fillna(train["Married"].mode()[0])
train["LoanAmount"]=train["LoanAmount"].fillna(train["LoanAmount"].median())
train["Gender"]=train["Gender"].fillna("O")
train["Dependents"]=train["Dependents"].fillna(0)
train["Self_Employed"]=train["Self_Employed"].fillna(train["Self_Employed"].mode()[0])
train["Loan_Amount_Term"]=train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median())
train["Credit_History"]=train["Credit_History"].fillna(train["Credit_History"].median())


# In[100]:


X=train.iloc[:,1:-1]
Y=train.iloc[:,12:]


# In[101]:


X["Gender"]=label_encoder.fit_transform(X["Gender"])
X["Married"]=label_encoder.fit_transform(X["Married"])
X["Education"]=label_encoder.fit_transform(X["Education"])
X["Self_Employed"]=label_encoder.fit_transform(X["Self_Employed"])
X["Property_Area"]=label_encoder.fit_transform(X["Property_Area"])
X["Dependents"].replace({"3+":"3"},inplace=True)
Y["Loan_Status"]=label_encoder.fit_transform(Y["Loan_Status"])


# In[102]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[103]:


cv = KFold(n_splits=10, random_state=42, shuffle=False)
rfm=RandomForestClassifier(n_estimators=100,min_samples_leaf = 3)
rf_scores=[]
for train_index, test_index in cv.split(X):
#     print("Train Index: ", train_index)
#     print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    rfm.fit(X_train, y_train)
    rf_scores.append(accuracy_score(y_test,rfm.predict(X_test))*100)


# In[104]:


statistics.mean(rf_scores)


# In[105]:


test_file=pd.read_csv("test.csv")


# In[106]:


test_file.head()


# In[107]:


test_file["Married"]=test_file["Married"].fillna(test_file["Married"].mode()[0])
test_file["LoanAmount"]=test_file["LoanAmount"].fillna(test_file["LoanAmount"].median())
test_file["Gender"]=test_file["Gender"].fillna("O")
test_file["Dependents"]=test_file["Dependents"].fillna(0)
test_file["Self_Employed"]=test_file["Self_Employed"].fillna(test_file["Self_Employed"].mode()[0])
test_file["Loan_Amount_Term"]=test_file["Loan_Amount_Term"].fillna(test_file["Loan_Amount_Term"].median())
test_file["Credit_History"]=test_file["Credit_History"].fillna(test_file["Credit_History"].median())


# In[108]:


X_tf=test_file.iloc[:,1:]


# In[109]:


X_tf["Gender"]=label_encoder.fit_transform(X_tf["Gender"])
X_tf["Married"]=label_encoder.fit_transform(X_tf["Married"])
X_tf["Education"]=label_encoder.fit_transform(X_tf["Education"])
X_tf["Self_Employed"]=label_encoder.fit_transform(X_tf["Self_Employed"])
X_tf["Property_Area"]=label_encoder.fit_transform(X_tf["Property_Area"])
X_tf["Dependents"].replace({"3+":"3"},inplace=True)


# In[110]:


X_tf_pred=rfm.predict(X_tf)


# In[111]:


loan_id=test_file.iloc[:,0:1]


# In[112]:


loan_id.head()


# In[113]:


sub=pd.concat([loan_id,pd.DataFrame(X_tf_pred)], axis=1, sort=False)


# In[114]:


sub=sub.rename(columns={0:"Loan_Status"})


# In[115]:


sub["Loan_Status"]=sub["Loan_Status"].replace(0.0,"N")
sub["Loan_Status"]=sub["Loan_Status"].replace(1,"Y")


# In[116]:


sub.to_csv("output.csv", index=False)

