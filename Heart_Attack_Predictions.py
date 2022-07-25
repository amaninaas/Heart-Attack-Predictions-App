# -*- coding: utf-8 -*-
"""
"""
#%% Module
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV

#%%
def cramers_corrected_stat(cmx):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(cmx)[0]
    n = cmx.sum()
    phi2 = chi2/n
    r,k = cmx.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%% Constant
CSV_PATH = os.path.join(os.getcwd(),'Dataset','heart.csv')

#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH)
#%% Step 2) Data Inspection
df.info()
des=df.describe().T
df.columns

cat_col = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
con_col = df.drop(labels=cat_col,axis=1).columns

# Categorical Visualization
for i in cat_col:
    print(i)
    plt.figure()
    sns.countplot(df[i])
    plt.show()

# Continous Visualization
for i in con_col: 
    plt.figure()
    sns.displot(df[i])
    plt.show()

df.boxplot(figsize=(20,5))

# To check NaNs
df.isna().sum()
# - From this we can see that, we can see that this dataset have no NaNs

#%% Step 3) Data Cleaning
# NaNs
# there are NaNs in caa (mask as 4) and thall (mask value 0)
df['caa'].replace(4, np.nan, inplace=True)
df['thall'].replace(0, np.nan, inplace=True)

# To check the NaNs
df.isna().sum()

# KNN Imputation
columns_names = df.columns
knn_i = KNNImputer()
df = knn_i.fit_transform(df) # return numpy array
df = pd.DataFrame(df) # to convert back into dataframe
df.columns = columns_names

#To check the if there is any duplicated data in the datasets
df.duplicated().sum() # From this data, there are 1 duplicated datasets

# Removing Duplicated
df = df.drop_duplicates()

#%% Step 4) Features selection
y = df['output']
selected_features = []

# print(df.corr)

# To check correlation between continous data vs categorical data
for i in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=-1),y)
    print(i)
    print(lr.score(np.expand_dims(df[i],axis=-1),y))
    if lr.score(np.expand_dims(df[i],axis=-1),y) >= 0.5:
        selected_features.append(i)

print(selected_features)
# From con_col, the features that are more than 0.5 are age,trtbps,
# chol,thalachh,oldpeak.

# To check correlation between categorical vs categorical data
for i in cat_col:
    print(i)
    cmx = pd.crosstab(df[i],y).to_numpy()
    print(cramers_corrected_stat(cmx))
    if cramers_corrected_stat(cmx) >= 0.4:
        selected_features.append(i)

print(selected_features)
# From cat_col, the features that have correlation more than 0.4 with y 
# are cp,exng,slp,caa,thall,output

# to visualize the correlation using heatmap
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='RdBu')

#%% Step 5) Data Preprocessing
# From Step 4, there are 10 selected features that are more than 0.4 correlation
# which are age,trtbps,chol,thalachh,oldpeak,cp,exng,slp,caa,thall.

df = df.loc[:,selected_features]
X = df.drop(labels='output',axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=123)

#%% Model Development
# KNN
pipeline_mms_knn = Pipeline([
                            ('Min_Max_Scalar', MinMaxScaler()),
                            ('KNN_Classifier', KNeighborsClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('KNN_Classifier', KNeighborsClassifier())
                            ]) # Pipeline([STEPS])

# RandomForest
pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scalar',MinMaxScaler()),
                            ('Forest_Classifier',RandomForestClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_rf = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Forest_Classifier',RandomForestClassifier())
                            ]) # Pipeline([STEPS])

#Logistic Regression
pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scalar', MinMaxScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ]) # Pipeline([STEPS])

pipeline_ss_lr = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ]) # Pipeline([STEPS])

# Decision Tree
pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scalar', MinMaxScaler()),
                            ('Tree_Classifier', DecisionTreeClassifier())
                            ]) # Pipeline([STEPS])

pipeline_ss_dt = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('Tree_Classifier', DecisionTreeClassifier())
                            ]) # Pipeline([STEPS])

# SVC
pipeline_mms_svc = Pipeline([
                            ('Min_Max_Scalar', MinMaxScaler()),
                            ('SVC_Classifier', SVC())
                            ]) # Pipeline([STEPS])

pipeline_ss_svc = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('SVC_Classifier', SVC())
                            ]) # Pipeline([STEPS])

pipelines = [pipeline_mms_knn,pipeline_ss_knn,pipeline_mms_rf,pipeline_ss_rf,
              pipeline_mms_lr,pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
              pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy = 0

for i,pipe in enumerate(pipelines):
    print(pipe.score(X_test, y_test))
    if pipe.score (X_test, y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test, y_test)
        best_pipeline = pipe

print('The best scaler and classifier for HAP app is {},with accuracy of {}'.
      format(best_pipeline.steps,best_accuracy))

# From Model Development steps,the best scaler and classifier for HAP app is 
# [('Standard_Scaler', StandardScaler()), ('Logistic_Classifier', LogisticRegression())]
# with accuracy of 0.8461538461538461

#%% GridSearchCV
# To check the best parameter of the best model.
pipeline_mms_lr = Pipeline([
                            ('Standard_Scaler', StandardScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ]) # Pipeline([STEPS])

grid_param = [{'Logistic_Classifier__random_state':[None,10,15],
                'Logistic_Classifier__tol':[1,2,3,5],
                'Logistic_Classifier__C':[1.0,3.0,5.0],
                'Logistic_Classifier__solver':['newton-cg','lbfgs','liblinear',
                                              'sag','saga'],
                'Logistic_Classifier__intercept_scaling': [1,2,3]
                }]

grid_search = GridSearchCV(pipeline_mms_lr,param_grid=grid_param,cv=5,
                            verbose=1, n_jobs=-1)
grid = grid_search.fit(X_train, y_train) 


print(grid.best_score_)
# The best score for grid is equal to 0.8529346622369879
print(grid.best_params_)
# The best params are {'Logistic_Classifier__C': 5.0, 
#                     'Logistic_Classifier__intercept_scaling': 1, 
#                     'Logistic_Classifier__random_state': None, 
#                     'Logistic_Classifier__solver': 'saga', 
#                     'Logistic_Classifier__tol': 5}
print(grid.best_estimator_)
# The best estimator is 
# Pipeline(steps=[('Standard_Scaler', StandardScaler()),
#                 ('Logistic_Classifier',
#                  LogisticRegression(C=5.0,
#                                     solver='saga',
#                                     tol=5))])

# To check the accuracy of the model
y_pred = grid.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

# Plotting heatmap 
labels = ['0','1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Print classification report
print(cr)

#%% Model Saving

BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(),'Models',
                                        'HAP_App_model.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(grid.best_estimator_,file)

