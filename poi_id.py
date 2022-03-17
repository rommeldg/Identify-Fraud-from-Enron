import sys
import pickle
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

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

poi = 0
for name in enron_data.values():
    if name['poi']:
        poi += 1
print("number of poi: ", poi)
print("number of person who is not poi: ", len(enron_data) - poi)    

#number of poi:  18
#number of person who is not poi:  128

# Convert dataset to panda dataframe for ea, then transpose 
import pandas as pd
import numpy as np
   
df_enron = pd.DataFrame(enron_data)
df_enron = df_enron.transpose()

# panda dataframe shape

df_enron.shape

#Out[5]: (146, 21)
   
'''
Dataset Information
The dataset has 146 datapoints and 21 features, with 128 non-POIs and 18 POIs. In addition, it contained real email messages between senior management (poi's and non-poi's). Therefore, we can explore this dataset, identify email patterns, and investigate any correlations between salary bonuses within senior management.
'''

#panda dataframe head
df_enron.head()

# dataset data type info, prior to data type conversion
df_enron.info()

# panda dataframe features value description 
df_enron.describe()


#I'll delete pandas dataframe features that I deemed unimportant from this exploration ('deferral_payments', loan_advances','restricted_stock_deferred','deferred_income','other','director_fees') and have more than 50% missing values ('NaN)

df_enron.drop(['deferral_payments', 'loan_advances','restricted_stock_deferred','deferred_income', 'other','director_fees'], axis=1, inplace=True)

#Then convert pandas dataframe features data type for 'salary', 'total_payments', 'bonus', 'total_stock_value' and 'exercised_stock_options' to 'Float64'.
df_enron["salary"] = df_enron.salary.astype(float)
df_enron["total_payments"] = df_enron.total_payments.astype(float)
df_enron["bonus"] = df_enron.bonus.astype(float)
df_enron["total_stock_value"] = df_enron.total_stock_value.astype(float)
df_enron["exercised_stock_options"] = df_enron.exercised_stock_options.astype(float)
df_enron["long_term_incentive"] = df_enron.long_term_incentive.astype(float)


#Pandas dataframe after converting some columns data type to float64.
df_enron.info()

print('Number of datapoints before outliers removal: ', len(enron_data)) 

#####################################################################################################################

### Task 2: Remove outliers


#I will remove any values that stand out from the sorted by names list below.
import pprint

pretty = pprint.PrettyPrinter()

names = sorted(enron_data.keys())

print('Enron employees sorted by last names')
pretty.pprint(names)

#After reviewing the names of employees from the sorted list of employees by last name above, I noticed two values that are not valid names. #These are 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL' and will also remove from the dataset.

enron_data.pop('THE TRAVEL AGENCY IN THE PARK',0)
enron_data.pop('TOTAL',0)

#I want to review the list of employees values for **'total payments'** and **'total stock'** and remove the ones with empty/Nan values from #both these features.

outliers =[]
for key in enron_data.keys():
    if  (enron_data[key]['total_payments']=='NaN') & (enron_data[key]['total_stock_value']=='NaN') :
        outliers.append(key)
print ("Enron employees outliers:",(outliers))

#After running a query on employees with null values on "Total Payments" and "Total Stock Values" features from the dataset, it returned #three employees with null values; therefore, I will remove them from the dataset. 'CHAN RONNIE', 'POWERS WILLIAM', and LOCKHART EUGENE E' #from the dataset.

enron_data.pop('CHAN RONNIE',0)
enron_data.pop('POWERS WILLIAM',0)
enron_data.pop('LOCKHART EUGENE E',0)

print('Number of people after outliers removal: ', len(enron_data)) 

#I will then list the names of POI's from the dataset.
df_enron[df_enron['poi'] == True]

#Then show the data statistics of these poi's.
df_enron[df_enron['poi'] == True].describe()

#Let's look into how many poi's are there with missing values and how many are there
df_enron[df_enron["poi"] == True].isnull().sum()


###############################################################################################################

### Task 3: Create new feature(s)

#I will be creating two new features to represent the message ratios of emails coming from poi's and message ratios of emails sent to poi's. #Then pass these new features to the SelectKBest function for feature selection.


def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    
    if poi_messages=='NaN' or all_messages=='NaN':
        fraction = 0.
    else:  
        fraction=float(poi_messages)/float(all_messages)

    return fraction


for name in enron_data:

    data_point = enron_data[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    
features_list2 = total_features
features_list2.remove('email_address')
features_list2 =  features_list2 + ['fraction_from_poi', 'fraction_to_poi']


fraction_to_poi =[enron_data[key]["fraction_to_poi"] for key in enron_data]
fraction_from_poi=[enron_data[key]["fraction_from_poi"] for key in enron_data]
poi=[enron_data[key]["poi"]==1 for key in enron_data]


def Second(elem):
    """ sorted second element
    """
    return elem[1]


import matplotlib.pyplot as plt
from feature_format import featureFormat

import matplotlib.pyplot as plt
from feature_format import featureFormat

def dict_to_list(key,normalizer):
    my_list=[]

    for i in enron_data:
        if enron_data[i][key]=="NaN" or enron_data[i][normalizer]=="NaN":
            my_list.append(0.)
        elif enron_data[i][key]>=0:
            my_list.append(float(enron_data[i][key])/float(enron_data[i][normalizer]))
    return my_list

### create two lists of new features
fraction_from_poi = dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi = dict_to_list("from_this_person_to_poi","from_messages")


### insert new features into data_dict
count=0
for i in enron_data:
    enron_data[i]["fraction_from_poi"]=fraction_from_poi[count]
    enron_data[i]["fraction_to_poi"]=fraction_to_poi[count]
    count +=1

    
new_features = ["poi", "fraction_from_poi", "fraction_to_poi"] 

    ### store to my_dataset for easy export below
my_dataset = enron_data

### Extract features and labels from dataset for local testing

from feature_format import targetFeatureSplit

data = featureFormat(my_dataset, features_list2, sort_keys = True)
labels, features = targetFeatureSplit(data)

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
selector = SelectKBest(f_classif, k = 13)
selector.fit(features, labels)
scores = zip(features_list2[1:], selector.scores_)
sorted_scores = sorted(scores, key = Second, reverse = True)
#pprint.pprint('SelectKBest scores: ')
pprint.pprint( sorted_scores)
all_features = POI_label + [(i[0]) for i in sorted_scores[0:20]]
#pprint.pprint( all_features)
SelectKBest_features = POI_label + [(i[0]) for i in sorted_scores[0:10]]
#pprint.pprint( 'KBest')
pprint.pprint( SelectKBest_features)
#print(my_dataset)
for emp in enron_data:
     for f in enron_data[emp]:
         if enron_data[emp][f] == 'NaN':
             # fill NaN values
             enron_data[emp][f] = 0
my_dataset = enron_data



################################################################################################

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# dataset using original features
from sklearn import preprocessing
data = featureFormat(my_dataset, SelectKBest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# dataset with new added features
SelectKBest_features = SelectKBest_features + ['fraction_from_poi', 'fraction_to_poi']
data = featureFormat(my_dataset, SelectKBest_features, sort_keys = True)
new_labels, new_features = targetFeatureSplit(data)
new_features = scaler.fit_transform(new_features)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def tune_params(grid_search, features, labels, params, iters = 80):
    """ given a grid_search and parameters list (if exist) for a specific model,
    along with features and labels list,
    it tunes the algorithm using grid search and prints out the average evaluation metrics
    results (accuracy, percision, recall) after performing the tuning for iter times,
    and the best hyperparameters for the model
    """
    acc = []
    pre = []
    recall = []
    
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predicts)] 
        pre = pre + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
    print ("accuracy: {}".format(np.mean(acc)))
    print ("precision: {}".format(np.mean(pre)))
    print ("recall: {}".format(np.mean(recall)))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))

#### 1.  SUPPORT VECTOR MACHINES

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
        
## SVM model evaluation with Original Features
## Precision = 0.21065476190476193
## Recall = 0.08765873015873016
## SVM model evaluation with New Features
## Precision = 0.15729166666666666
## Recall = 0.057668650793650786


#### 2.DECISION TREE

from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

print("Decision Tree model evaluation with Original Features")
tune_params(dt_grid_search, features, labels, dt_param)
print("Decision Tree model evaluation with New Features")
tune_params(dt_grid_search, new_features, new_labels, dt_param)

## Decision Tree model evaluation with Original Features
## Precision = 0.33407151875901875
## Recall = 0.3296875
## Decision Tree model evaluation with New Features
## Precision = 0.3017135642135642
## Recall = 0.2996577380952381

        

#### 3. NAIVE BAYES

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(estimator = nb_clf, param_grid = nb_param)
print("Naive Bayes model evaluation with Original Features")
tune_params(nb_grid_search, features, labels, nb_param)
print("Naive Bayes model evaluation with new features")
tune_params(nb_grid_search, new_features, new_labels, nb_param)

## Naive Bayes model evaluation with Original Features**<br>
## Precision = 0.3991815476190476*<br>
## Recall = 0.33036706349206346*<br>
## Naive Bayes model evaluation with New Features**<br>
## Precision = 0.35005005411255413*<br>
## Recall = 0.31953373015873016*


####  4. Random Forest

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)
rf_param = {}
rf_grid_search = GridSearchCV(estimator = rf_clf, param_grid = rf_param)

print("Random Forest model evaluation")
tune_params(rf_grid_search, features, labels, rf_param)
print("Random Forest model evaluation with New Features")
tune_params(rf_grid_search, new_features, new_labels, rf_param)

##################################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

from sklearn import preprocessing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels1, new_features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
new_features = scaler.fit_transform(new_features)

from sklearn.feature_selection import SelectKBest, f_classif
def feature_selection(nb_features,features, labels):
    selector = SelectKBest(f_classif, k=nb_features)
    selector.fit(features, labels)
    return selector

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

def test_classifier(clf, labels, features, nb_features, folds = 1000):
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    precision=0
    recall=0
    f1=0
    f2=0
    for train_idx, test_idx in cv.split(features, labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
            
        #Selection of the best K features   
       # selector=feature_selection(nb_features,features_train, labels_train)
        selector=feature_selection(nb_features,features_train, labels_train)
        features_train_transformed = selector.transform(features_train)
        features_test_transformed  = selector.transform(features_test)   
            
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train_transformed, labels_train)
        predictions = clf.predict(features_test_transformed)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                break
   
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    except:
        None
    return precision,recall,f1,f2


#### Decission Tree Classifier  
#  Make a note of the different metrics

from sklearn import tree

nb_features_orig=len(new_features[1])

precision_result=[]
recall_result=[]
f1_result=[]
f2_result=[]
nb_feature_store=[]

dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}

# calculate 
for nb_features in range(1,nb_features_orig+1):
    #Number of neighbours
           
        #classifier
        clf = tree.DecisionTreeClassifier()
        #Cross-validate then calculate it's precision and recall metrics
        precision,recall,f1,f2=test_classifier(clf, labels1, new_features,nb_features, folds = 1000)        
        # Note each evaluation metrics 
        precision_result.append(precision)
        recall_result.append(recall)     
        f1_result.append(f1)
        f2_result.append(f2)
        nb_feature_store.append(nb_features)
    
import pandas as pd
result=pd.DataFrame([nb_feature_store,precision_result,recall_result,f1_result,f2_result]).T
result.columns=['nb_feature','precision','recall','f1','f2']
result.head()


#### Gaussian Naive Bayes (GaussianNB)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
nb_features_orig=len(new_features[1])

precision_result=[]
recall_result=[]
f1_result=[]
f2_result=[]
nb_feature_store=[]

#Classifier
clf=GaussianNB()
#calculate the evaluation metrics for k best number of features selected in the model.
for nb_features in range(1,13):    
    # Cross-validation and calculate precision and recall metrics
    precision,recall,f1,f2=test_classifier(clf, labels1, new_features, nb_features, folds = 1000)    
    # Note each evaluation metrics               
    precision_result.append(precision)
    recall_result.append(recall)     
    f1_result.append(f1)
    f2_result.append(f2)
    nb_feature_store.append(nb_features)

import pandas as pd
result=pd.DataFrame([nb_feature_store,precision_result,recall_result,f1_result,f2_result]).T
result.columns=['nb_feature','precision','recall','f1','f2']
result.head(10)  
    
##################################################################################################

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from tester import dump_classifier_and_data

### Task 6: Dump your classifier, dataset, and features_list 
features_list = total_features
dump_classifier_and_data(clf, my_dataset, features_list)