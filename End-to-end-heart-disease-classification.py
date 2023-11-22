#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease Using Machine Learning
# 
# Name:Suprasanna V Gunaga.
# 
# Stream:Information Science and Engineering, RNS Institue Of Technology,Bengaluru.
# 
# Year: 3rd.
# 
# This notebook looks into using various Python-based machine learning and data science
# libraries in an attempt to build a machine learning model capable of predicting whether or
# not someone has heart disease based on their medical attributes

# # We are going to take the following approach:
#     1.Problem definition
#     2.Data
#     3.Evaluation
#     4.Features
#     5.Modelling
#     6.Experimentation

# # Data Dictionary
# 
# *age: Displays the age of the individual.
# 
# *sex: Displays the gender of the individual using the following format 
# 
# *cp- Chest-pain type: displays the type of chest-pain experienced by the individual 
# 
# *trestbps- Resting Blood Pressure:
# 
# *chol- Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)
# 
# *fbs- Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
# 
# *restecg- Resting ECG : displays resting electrocardiographic results
# 
# *thalach- Max heart rate achieved : displays the max heart rate achieved by an individual.
# 
# *exang- Exercise induced angina : 1 = yes 0 = no
# 
# *oldpeak- ST depression induced by exercise relative to rest: displays the value which is an integer or float.
# 
# *slope- Slope of the peak exercise ST segment 
# 
# *ca- Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
# 
# *thal : Displays the thalassemia 
# 
# *target : Displays whether the individual is suffering from heart disease or not  if have=1,not=0

# In[62]:


#Regular EDA(exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#Model from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Model Evaluations
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay



# In[8]:


#Loading the data
df=pd.read_csv("CSV files/heart-diseasep1.csv")
df.shape #rows,column


# In[9]:


#Exploring data
df.head()


# In[10]:


#Lets find out how many of each class there
df["target"].value_counts()


# In[18]:


df["target"].value_counts().plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"]);


# In[12]:


df.info()


# In[13]:


df.isna().sum()


# In[14]:


df.describe()


# Heart Disease frequency according to Sex

# In[15]:


df.sex.value_counts()


# In[16]:


#Compare target column with sex column
pd.crosstab(df.target,df.sex)


# In[22]:


pd.crosstab(df.target,df.sex).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"]);
plt.title("heart Disease Frequency for Sex")
plt.xlabel("0=No Disease,1=Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
# plt.xtricks()


# # Age vs Max Heart Rate for Heart Disease

# In[27]:


#Create another figure
plt.figure(figsize=(10,6))

#scatter with postivie example
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c="salmon");

#scatter with negative example
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c="lightblue");

# Adding some helpful info
plt.title("Heart Disease in function of Age and Max heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease","No Disease"]);


# In[34]:


df.age.plot.hist(figsize=(7,4));


# # Heart Disease frequency per Chest Pain Type
# 
# cp-chest pain type
# 
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 
# 1: Atypical angina: chest pain not related to heart
# 
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 
# 3: Asymptomatic: chest pain not showing signs of disease

# In[35]:


pd.crosstab(df.cp,df.target)


# In[36]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])

plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease","Disease"])
plt.xticks(rotation=0);


# In[37]:


#Making a correlation matrix
df.corr()


# In[40]:


corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
              annot=True,
              linewidths=0.5,
              fmt=".2f",
              cmap="YlGnBu");


# We are going to try 3 different machine learning models;
# 1.logistic Regression
# 2.K-Nearest Neighbours Classifier
# 3.Random Forest Classifier

# # Modelling

# In[44]:


#splitting data into x and y
x=df.drop("target",axis=1)
y=df["target"]

#Split data into train and test sets
np.random.seed(42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



# In[41]:


#putting models in dictionary
models={"Logistic Regression": LogisticRegression(),
       "KNN":KNeighborsClassifier(),
       "Random Forest":RandomForestClassifier()}

#Create a function to fit score models

def fit_and_score(models,x_train,x_test,y_train,y_test):
    
    #set random seed
    np.random.seed(42)
    #Make a dictionary to keep model scores
    model_scores={}
    #Loop through models
    for name,model in models.items():
        #Fit the model to the data
        model.fit(x_train,y_train)
        #Evaluate the model and append its score to model_scores
        model_scores[name]=model.score(x_test,y_test)
    return model_scores
        


# In[45]:


model_scores=fit_and_score(models=models,
                          x_train=x_train,
                          x_test=x_test,
                          y_train=y_train,
                          y_test=y_test)
model_scores


# # Model comparision

# In[47]:


model_compare=pd.DataFrame(model_scores,index=["accuracy"])
model_compare.T.plot.bar(figsize=(10,6));

#Improving Model

*Hyperparameter tuning
*Feature importance
*Confusion matrix
*Cross-validation
*Precision
*Recall
*F1 score
*Classification report
*ROC curve
*Area under the curve
# In[50]:


#Tune the KNN

train_scores=[]
test_scores=[]

#Creating a list of different values for n_neighbors
neighbors=range(1,21)

#Setup KNN instance
knn=KNeighborsClassifier()

#Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    #Fit the algorithm
    knn.fit(x_train,y_train)
    
    #Update the training scores list
    train_scores.append(knn.score(x_train,y_train))
    
    #Update the test scores list
    test_scores.append(knn.score(x_test,y_test))


# In[51]:


train_scores


# In[52]:


test_scores


# In[55]:


plt.plot(neighbors, train_scores,label="Train score")
plt.plot(neighbors, test_scores,label="Test score")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# # Hyperparameter tuning with Randomized searchCV
# 
# Tuning 
# 
# 1.LogisticRegression()
# 
# 
# 2.RandomForestClassifier()
# 

# In[71]:


#Creating a hyperparameter grid for LogisticRegression
log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":["liblinear"]}

#Creating a hyperparameter grid for RandomForestClassifier
rf_grid={"n_estimators":np.arange(10,1000,50),
         "max_depth":[None,3,5,10],
         "min_samples_split":np.arange(2,20,2),
         "min_samples_leaf":np.arange(1,20,2)}


# Now we have hyperparameter grids setup for each of our models,lets tune them using RandomizedsearchCv

# In[66]:


#Tune LogisticRegression
np.random.seed(42)

#Setup random hyperparameter search for LogisticRegression
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                             param_distributions=log_reg_grid,
                             cv=5,
                             n_iter=20,
                             verbose=True)
#Fitting random hyperparamter seach model for Logisticregression
rs_log_reg.fit(x_train, y_train)


# In[67]:


rs_log_reg.best_params_


# In[68]:


rs_log_reg.score(x_test,y_test)


# In[74]:


#RandomForestClassifier

np.random.seed(42)

#Setup random hyperparameter search for RandomForestClassifier
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                             param_distributions=rf_grid,
                             cv=5,
                             n_iter=20,
                             verbose=True)
#Fitting random hyperparamter seach model for RandomForestClassifier
rs_rf.fit(x_train, y_train)


# In[75]:


#Finding best hyperparameters
rs_rf.best_params_


# In[77]:


#Evaluate the randomized search RandomForestClassifier model
rs_rf.score(x_test,y_test)


# # Hyperparameter Tuning with GridSearchCv
# Since our LogisticRegression model provides the scores so far, now we can check
# using GridSearchCV..

# In[81]:


#Different hyperparameter for our Logisticregression model
log_reg_grid={"C": np.logspace(-4,4,30),
             "solver":["liblinear"]}

#Set grid hyperparameter search for LogisticRegression
gs_log_reg=GridSearchCV(LogisticRegression(),
                       param_grid=log_reg_grid,
                       cv=5,
                       verbose=True)
#Fit grid hyperparameter search model
gs_log_reg.fit(x_train,y_train)


# In[82]:


#Check the best hyperparameter
gs_log_reg.best_params_


# In[83]:


#Evaluate the grid search LogisticRegression model
gs_log_reg.score(x_test,y_test)


# # Evaluating our tuned machine learning classifier,beyond accuracy
# 
# *ROC curve and AUC score
# 
# *Confusion matrix
# 
# *Classifier report
# 
# *Precision
# 
# *Recall
# 
# *F1-score
# 
#  To make comparisons and evaluate our trained model, first we need to make predictions

# In[84]:


#Make prediction with tuned model
y_preds=gs_log_reg.predict(x_test)


# In[85]:


y_preds


# In[86]:


y_test


# In[93]:


#Plot ROc curve and calculate and calculate AUC metric
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay(gs_log_reg,x_test,y_test)


# In[91]:


# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size
 
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label") # predictions go on the x-axis
    plt.ylabel("True label") # true labels go on the y-axis 
    
plot_conf_mat(y_test, y_preds)


# In[94]:


print(classification_report(y_test,y_preds))


# # Calculating evaluation metrics using cross-validation
# 
# We are going to calculate accuaracy,precision,recall, and f1-score of our model cross-validation
# and to do so we will be using cross_val_score()

# In[95]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[96]:


# Create a new classifier with best parameters
clf =LogisticRegression(C=0.20433597178569418,
                       solver="liblinear")


# In[100]:


# Cross-validated accuaracy
cv_acc=cross_val_score(clf,
                      x,
                      y,
                      cv=5,
                      scoring="accuracy")
cv_acc


# In[104]:


cv_acc=np.mean(cv_acc)
cv_acc


# In[105]:


# Cross-validated precision
cv_precision=cross_val_score(clf,
                      x,
                      y,
                      cv=5,
                      scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision


# In[106]:


# Cross-validated recall
cv_recall=cross_val_score(clf,
                      x,
                      y,
                      cv=5,
                      scoring="recall")
cv_recall=np.mean(cv_recall)
cv_recall


# In[108]:


# Cross-validated f1-score
cv_f1=cross_val_score(clf,
                      x,
                      y,
                      cv=5,
                      scoring="f1")
cv_f1=np.mean(cv_f1)
cv_f1


# In[111]:


# Visualize cross-validated metrics
cv_metrics=pd.DataFrame({"Accuracy":cv_acc,
                        "Precision":cv_precision,
                        "Recall":cv_recall,
                        "F1": cv_f1},
                       index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics",figsize=(10,6),legend=False);


# # Feature Importance
# 
# Feature importance is another as asking, "which features contributed most to the outcomes of the model and how did they contribute?"
# 
# Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# Let's find the feature importance for our LogisticRegression model...

# In[112]:


clf =LogisticRegression(C=0.20433597178569418,
                       solver="liblinear")
clf.fit(x_train,y_train);


# In[113]:


#checking coef_
clf.coef_


# In[114]:


# Maching coefs of features to columns
features_dict=dict(zip(df.columns,list(clf.coef_[0])))
features_dict


# # Visualize features importance
# 

# In[118]:


features_df=pd.DataFrame(features_dict,index=[0])
features_df.T.plot.bar(title="Features Importance",legend=False);


# 

# In[ ]:




