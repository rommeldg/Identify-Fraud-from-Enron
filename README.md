# Identify-Fraud-from-Enron
Udacity Data Analyst Nanodegree Project - Machine Learning 
* #### Rommel DeGuzman

## Enron Scandal Summary



In early December 2001, innovative energy company Enron Corporation, a darling of Wall Street investors with $63.4 billion in assets, went bust. It was the largest bankruptcy in U.S. history. Some of the corporation’s executives, including the CEO and chief financial officer, went to prison for fraud and other offenses. Shareholders hit the company with a 40 billion dollar lawsuit, and the company’s auditor, Arthur Andersen, ceased doing business after losing many of its clients.

It was also a black mark on the U.S. stock market. At the time, most investors didn’t see the prospect of massive financial fraud as a real risk when buying U.S.-listed stocks. “U.S. markets had long been the gold standard in transparency and compliance,” says Jack Ablin, founding partner at Cresset Capital and a veteran of financial markets. 

The company’s collapse sent ripples through the financial system, with the government introducing a set of stringent regulations for auditors, accountants and senior executives, huge requirements for record keeping, and criminal penalties for securities laws violations. In turn, that has led in part to less choice for U.S. stock investors, and lower participation in stock ownership by individuals.

#### 1. Goal

* This project aims to look into the Enron dataset using a machine learning algorithm to identify the POI (Persons of Interest) and non-POI employees based on the public Enron financial and email corpus. Enron was an energy company and the darling of Wall Street investors for years until it went bust due to fraud and other offenses. 


### Understanding the Dataset and Question

#### 2. Dataset Exploration

> - **Data Exploration (related lesson: "Datasets and Questions")** - Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:<br>
       - total number of data points<br>
       - allocation across classes (POI/non-POI)<br>
       - number of features used<br>
       - are there features with many missing values? etc.<br>
> - **Outlier Investigation (related lesson: "Outliers")** - Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

import sys
import pickle
sys.path.append("../tools/")
import numpy as np


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use. Features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','bonus', 'email_address', 'total_stock_value', 'expenses', 'other', 'long_term_incentive',
                 'restricted_stock','total_stock_value', 'exercised_stock_options','total_payments', 'deferred_income'] 

financial_features = ['salary', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', ] 

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi'] 
           
POI_label = ['poi']
                 
total_features = features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    enron_data = pickle.load(data_file) 

* #### Dataset Information


> The dataset has **146** datapoints and **21** features, with **128** non-POIs and **18** POIs. In addition, it contained real email messages between senior management (poi's and non-poi's). Therefore, we can explore this dataset, identify email patterns, and investigate any correlations between salary bonuses within senior management.

#panda dataframe head

df_enron.head()

# dataset data type info, prior to data type conversion

df_enron.info()

# panda dataframe features value description 

df_enron.describe()

* **I'll delete pandas dataframe features that I deemed unimportant from this exploration ('deferral_payments', loan_advances','restricted_stock_deferred','deferred_income','other','director_fees') and have more than 50% missing values ('NaN)**

df_enron.drop(['deferral_payments', 'loan_advances','restricted_stock_deferred','deferred_income', 'other','director_fees'], axis=1, inplace=True)

* **Then convert pandas dataframe features data type for 'salary', 'total_payments', 'bonus', 'total_stock_value' and 'exercised_stock_options' to 'Float64'.**

df_enron["salary"] = df_enron.salary.astype(float)
df_enron["total_payments"] = df_enron.total_payments.astype(float)
df_enron["bonus"] = df_enron.bonus.astype(float)
df_enron["total_stock_value"] = df_enron.total_stock_value.astype(float)
df_enron["exercised_stock_options"] = df_enron.exercised_stock_options.astype(float)
df_enron["long_term_incentive"] = df_enron.long_term_incentive.astype(float)

* **Pandas dataframe after converting some columns data type to float64.**

df_enron.info()

>  After reviewing the names of employees from the sorted list of employees by last name above, I noticed two values that are not valid names.  These are **'THE TRAVEL AGENCY IN THE PARK'** and **'TOTAL'** and will also remove from the dataset.

enron_data.pop('THE TRAVEL AGENCY IN THE PARK',0)
enron_data.pop('TOTAL',0)

*  I want to review the list of employees values for **'total payments'** and **'total stock'** and remove the ones with empty/Nan values from both these features.

outliers =[]
for key in enron_data.keys():
    if  (enron_data[key]['total_payments']=='NaN') & (enron_data[key]['total_stock_value']=='NaN') :
        outliers.append(key)
print ("Enron employees outliers:",(outliers))

### Optimize Feature Selection/Engineering

> - **Create new features (related lesson: "Feature Selection")** - At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.<br>
> - **Intelligently select features (related lesson: "Feature Selection")** - Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.
> - **Properly scale features (related lesson: "Feature Scaling")** - If algorithm calls for scaled features, feature scaling is deployed.

* **I will be creating two new features to represent the message ratios of emails coming from poi's (fraction_from_poi) and message ratios of emails sent to poi's (fraction_to_poi). Then pass these new features to the SelectKBest function for feature selection.**


* ***These two new features will represent the ratios of the emails from poi (fraction_from_poi) to this person divided with all the other emails sent to person. And ratios of emails from this person to poi (fraction_to_poi) divided with all the emails from this person.***


* **Univariate feature selection works best through selecting the best features from a statistical tests. I utilized an automated feature selection function named SelectKBest.  Tuning the parameter k (number of features) while also tuning the parameters of machine learning algorithm when implementing cross-validation.  Selecting all 21 features from the dataset while tuning the parameters of the machine learning can result to overfitting.**


* ### Feature Scaling

Some of these features have different units and significant values and would be transformed by using sklearn **MinMaxScaler** to a given range of between **0** and **1**.


*  **I am utilizing an automated feature function SelectKBest from sklearn to select the best K features.**
# dataset with new added features
SelectKBest_features = SelectKBest_features + ['fraction_from_poi', 'fraction_to_poi']
data = featureFormat(my_dataset, SelectKBest_features, sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

 #### 1.  Support Vector Machines
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm


svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)


print("SVM model evaluation with Original Features")
tune_params(svm_grid_search, features, labels, svm_param)
print("SVM model evaluation with New Features")
tune_params(svm_grid_search, new_features, new_labels, svm_param)

### Pick and Tune an Algorithm
* I tried three different algorithms and have decided to use **Naive Bayes** since it got the highest evaluation score. I  also tried **"SVM"** and **"Decision Tree"****. These algorithms all showed higher accuracy scores and are probably not the best metric to use.

* Since most algorithms have multiple default values, tuning the classifier's specific parameters can help optimize its performance; otherwise, the data model can either be overfitting or underfitting.  So, for example, we adjust SVM's hyperparameter '**kernel**', '**gamma**', and '**C**' to achieve the best possible performance.  This is called hyperparameter optimization.  It is an essential step in machine learning before the presentation.

* I have used sklearn's **GridSearchCV** library function as parameter tuning. They are used in  **SVM** and **Decision Tree** algorithm.  **GridSearchCV** implements a fit and score method and evaluate a model in each specified parameter combination.

### Validate and Evaluate
* We use validation to evaluate the classifier using its training and testing dataset. We use it to measure its reliability and accuracy.  If we train and test the classifier with the same data, it will yield overfitting results, so validation is essential. I will use StratifiedShuffleSplit to split the data between the training and testing datasets. This will guarantee that the classes are randomly selected and correctly allocated.


### Evaluation metrics

I will use the two evaluation metrics  **Precision and Recall**, which is used best in measuring prediction success with highly imbalanced classes.  When retrieving information, **precision** measures the relevance of its result, while **recall** measures how accurately relevant are its results.

F1 scores measures the weighted average of precision and recall.  

If the precision score is **0.51**, then it means there is a  **51%** chance that the predicted POIs are truly POIs

If the recall score is **0.42**, then it means there is a **42%** chance that the POIs were identified correctly.

from tester import dump_classifier_and_data

### Task 6: Dump your classifier, dataset, and features_list 
features_list = total_features
dump_classifier_and_data(clf, my_dataset, features_list)







