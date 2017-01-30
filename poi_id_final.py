#!/usr/bin/python

import sys

sys.path.append("../tools/")
sys.path.append("../feature_selection/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from fraction_poi import submitDict

import pickle
import csv
import pprint as pp
import pandas as pd
from time import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


#################################
###### PREPARE VARIABLES #############

### Set hard-coded variables

random_state = 888
scoring = 'f1_weighted' # scoring type used for cross-validation, irrelevant for 'simple' mode
iter_count = 0

### OPTION A: VARIABLES FOR SINGLE MODEL

### NOTE: The same-named variables are also used for testing scenarios.
### To use this script for testing, comment out these variables, and 
### uncomment the section that follows.

kind = 'simple' # feature selection + param tuning
use_f = 'uncorr' # use features that have > .7 correlations 
incl_new = 'true' # include 4 new features
data_s = 'train' # split into train & test data sets

# assemble variables into instruction list
instr = {}
keys = ('kind', 'feature_list', 'new_features', 'dataset')
vals = (kind, use_f, incl_new, data_s)
# create dict of instruction
for i,v in enumerate(keys):
    instr[v] = vals[i] 

# put dict into list
instr_list = [instr]

# tracking iterations here too so it doesn't break 
# iteration counts used for the looping test script
iterations = len(instr_list)
iter_count = 0

# list of algorithms to be used with the specified classifer kind/mode
algos = [DecisionTreeClassifier(random_state=random_state)]


### OPTION B: VARIABLES FOR TESTING MULTIPLE MODELS

### NOTE: The same-named variables are also used for creating a single model.
### To use this script for testing, uncomment out this section, 
### and comment out the section above

'''
# variables used for each iteration
keys = ('kind', 'feature_list', 'new_features', 'dataset')

# values for each variable
kind = ('simple', 'fs', 'pt', 'fspt')

# feature lists
use_f = ('all', 'uncorr', '>50%')

# include new features?
incl_new = ('true','false')

# use complete dataset, or split into train & test?
data_s = ('full', 'train')

# create different permutations as tuples
vals = [(a, b, c, d) for a in kind for b in use_f for c in incl_new for d in data_s]

# create list of instruction dictionaries
instr_list = [dict(zip(keys, i)) for i in vals]
iterations = len(instr_list)
iter_count = 0

# algorithms to use for testing
algos = [GaussianNB(),
         DecisionTreeClassifier(random_state=random_state),
         AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
         RandomForestClassifier(random_state=random_state)]
'''

#############################################
############  PREPARE DATA     #############
#############################################

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers

# remove total row & near-empty rows
outlier_list = ["TOTAL","THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E"]
map(data_dict.pop, outlier_list)

# correct processing errors
# see https://discussions.udacity.com/t/sanjay-bhatnagars-data/180305
sanjay = data_dict['BHATNAGAR SANJAY']

sanjay['expenses'] = 137864
sanjay['total_payments'] = 137864
sanjay['other'] ='NaN'
sanjay['director_fees'] = 'NaN'
sanjay['exercised_stock_options'] = 15456290
sanjay['total_stock_value'] = 15456290
sanjay['restricted_stock_deferred'] = -2604490
sanjay['restricted_stock'] = 2604490

robert = data_dict['BELFER ROBERT']

robert['deferred_income'] = -102500
robert['expenses'] = 3285
robert['director_fees'] = 102500
robert['total_payments'] = 3285
robert['deferral_payments'] = 'NaN'

robert['restricted_stock'] = 44093
robert['restricted_stock_deferred'] = -44093
robert['total_stock_value'] = 'NaN'
robert['exercised_stock_options'] = 'NaN'


### Get list of new features
submit_dict = submitDict()
map(submit_dict.pop, outlier_list) # remove outliers

# add a bunch of new features dynamically
new_features = submit_dict.values()[0].keys()
for name in submit_dict:
    for i in range(0,len(new_features)):
        #for i in range(0,1):
        nf = new_features[i]
        data_dict[name][nf] = submit_dict[name][nf]

        
        
        
#############################################
############ HELPER FUNCTIONS  ##############
#############################################

def printFeatureList(clf, features, labels, features_list):
    f = features_list[1:]
    clf.fit(features, labels)
    f_imp = clf.feature_importances_
    f_imp_list = pd.Series(f_imp, index = f)
    return f_imp_list
    
def printFeatureScores(select, features_list):
    f = features_list[1:]
    ss = select.scores_
    f_scores = pd.Series(ss, index = f)
    return f_scores
    
def getAlgoName(clf):
    algo = str(clf)
    name, _ = str(clf).split("(",1)
    return name

def getFeatureScores(features, labels):
    select = SelectKBest(f_classif)
    select.fit(features, labels) 
    f = features_list[1:]
    f_scores = pd.Series(select.scores_, index = f).order(ascending=False)
    return f_scores

 ## Create train & test data

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

def createShuffleTrainTest(features, labels, folds):
    shuffle = StratifiedShuffleSplit(labels, folds, random_state = 123)
    for train_idx, test_idx in shuffle: 
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
    return features_train, features_test, labels_train, labels_test



#############################################
############ CLASSIFIER FUNCTIONS  ##########
#############################################


### OOTB defaults, AKA 'simple' kind
def simpleClassifier(algos, x, y):
    print "=== FUNCTION: simpleClassifier ==="
    algo_names = []
    perf_dict = {}
    fi_dict = {}
    results = []
    #feat_length = len(features.columns)-1
    for i in range(0, len(algos)):
        clf = algos[i]
        clf.fit(x, y)
        print clf
        
        # get classifier name
        algo_name = getAlgoName(clf)
        algo_names.append(algo_name)
        
        # get feature importances
        #feat_imp = printFeatureList(clf, features_list, clf_name)
        #fi_dict[clf_name] = feat_imp
        
        # validate & collect results
        precision, recall, f1, f2, accuracy = \
        (test_classifier(clf, my_dataset, features_list, folds=100))
        
        algo_results = [precision, recall, f1, f2, accuracy]
        results.append(algo_results)
        
    results_df = pd.DataFrame(results, index = algo_names, columns = \
                              ['precision', 'recall','f1', 'f2', 'accuracy'])
    #fi_df = pd.DataFrame(fi_dict, index = features_list[1:])
    print "== SUMMARY =="
    #print "Number of features: " + str(len()
    print results_df
    #print fi_df    


    
# SelectKBest feature selection, AKA 'fs' kind 
def featureSelectCLF(features, labels, algo):
    # define pipeline variables
    scaler = MinMaxScaler()
    select = SelectKBest(f_classif)

    # create pipeline
    pipe = Pipeline([
            ('scaler', scaler),
            ('select', select),
            ('algo', algo)])

    # define params
    params = {'select__k': [i for i in range(1,len(features[0]+1))]}
            
    # perform grid search
    t0=time()
    clf_grid = GridSearchCV(pipe, params, scoring=scoring, cv=cv)
    clf_grid.fit(features, labels)
    print "training time:", round(time()-t0, 3), "s"

    clf = clf_grid.best_estimator_
    f_scores = getFeatureScores(features, labels)
    return clf, f_scores



### Parameter tuning only, AKA 'pt' kind
def paramTuneCLF(features, labels, algo):
    # build pipeline
    pipe = Pipeline([('algo', algo)])

    # define params
    if str(algo).startswith('GaussianNB'):
        params = {}

    elif str(algo).startswith('AdaBoostClassifier'):
        params = {'algo__algorithm': ['SAMME'], #SAMME.R isn't compatible with DT?
                  'algo__n_estimators': [25,50,75,100],
                  'algo__learning_rate': [0.25,0.5,0.75,1.0]
         }  

    #elif str(algo).startswith('DecisionTreeClassifier'):
    else:
        params = {'algo__criterion': ['gini','entropy'],
                  'algo__max_depth':range(1,10),
                  'algo__class_weight': ['balanced', {0:1,1:1}, {0:100,1:1}, {0:1,1:100}]
         }
        
    # perform grid search
    t0=time()
    clf_grid = GridSearchCV(pipe, params, scoring=scoring, cv=cv)
    clf_grid.fit(features, labels)
    print "training time:", round(time()-t0, 3), "s"

    clf = clf_grid.best_estimator_
    return clf



### to determine optimal k values for each algo
def FSLoop2(algos, features, labels):
    print "=== FUNCTION: FSLoop2 ==="
    algo_names = []
    perf_dict = {}
    fi_dict = {}
    results = []
    
    for i in range(0, len(algos)):
        algo = algos[i]
        
        # get classifier name
        algo_name = getAlgoName(algo)
        algo_names.append(algo_name)
        
        # to generate feature importances   
        algo.fit(features, labels) 
        
        for k in range(1, len(features[0])+1):
            # set up pipeline
            scaler = MinMaxScaler()
            select = SelectKBest(f_classif, k)
            
            
            # fit pipeline
            clf = Pipeline([('scaler', scaler),('select', select),('algo', algo)]) 
            #clf = Pipeline([('select', select),('algo', algo)]) 
            t0=time()
            clf.fit(features, labels)
            train_time = round(time()-t0, 3)
            
            # cross-validate
            precision, recall, f1, f2, accuracy = \
            (test_classifier(clf, my_dataset, features_list, folds=100))
        
            # collect results
            k_results = [algo_name, k, train_time, precision, recall, f1, f2, accuracy]
            results.append(k_results)
            
        print "* %s processing done *" % algo_name
    
    # convert results list to dataframe
    results_fs = pd.DataFrame(results, columns = \
                              ['clf_name','# features', 'train time', 'precision', 'recall','f1', 'f2', 'accuracy'])
        
    return results_fs




### Feature selection + param tuning classifier, AKA 'fspt' kind
def featureSelectParamTuneCLF(features, labels, algo, k):
    print "\r\n=== FUNCTION: featureSelectParamTuneCLF ==="
    # build pipeline
    scaler = MinMaxScaler()
    select = SelectKBest(f_classif, k)
    
    pipe = Pipeline([
            ('scaler', scaler),
            ('select', select),
            ('algo', algo)])

    # define params
    if str(algo).startswith('GaussianNB'):
        params = {}

    elif str(algo).startswith('AdaBoostClassifier'):
        params = {'algo__algorithm': ['SAMME'], #SAMME.R isn't compatible with DT?
                  'algo__n_estimators': [25,50,75,100],
                  'algo__learning_rate': [0.25,0.5,0.75,1.0]
         }  

    else:
        params = {'algo__criterion': ['gini','entropy'],
                  'algo__max_depth':range(1,10),
                  'algo__class_weight': ['balanced', {0:1,1:1}, {0:100,1:1}, {0:1,1:100}]
         }
        
    # perform grid search
    t0=time()
    clf_grid = GridSearchCV(pipe, params, scoring=scoring, cv=cv)
    clf_grid.fit(features, labels)
    print "training time:", round(time()-t0, 3), "s"

    clf = clf_grid.best_estimator_
    f_scores = getFeatureScores(features, labels)
    return clf, f_scores



### generates the CLF according to the 'mode' specified in the instructions
def createCLF(kind, features, labels, algos):
    print "\r\n=== RUNNING createCLF " + "="*50 + "\n"
    print "\rCLF FUNCTION MODE: %r" % kind
    print "SCORING: %r" % scoring
    print "# FEATURES: %r" % num_f
    algo_names = []
    perf_dict = {}
    fi_dict = {}
    results = []
    
    if kind == 'fspt':
        # find optimal K values for each algo
        print "Finding optimal k values for feature selection..."
        results_fs = FSLoop2(algos, features, labels)
        max_f1 = results_fs.groupby(['clf_name'])['f1'].max() 
        locs = results_fs.loc[results_fs['f1'].isin(max_f1)]
        
        print "* Optimal k values for each algo: *"
        print "\r"
        print locs[['clf_name', '# features', 'f1', 'precision', 'recall']]
      
    
    for i in range(0, len(algos)):
        
        # get algorithm name
        algo = algos[i]
        algo_name = getAlgoName(algo)
        algo_names.append(algo_name)
        print "\nALGORITHM: ", algo_name
         
        # define and fit classifier
        if kind == 'simple':
            clf = algo
        
        elif kind == 'fs':
            clf, f_scores = featureSelectCLF(features, labels, algo)            
        
        elif kind == 'pt':
            clf = paramTuneCLF(features, labels, algo)
            
        elif kind == 'fspt':
            # get optimal # of features for this algo
            k_row = locs[locs['clf_name'] == algo_name]
            #k = int(k_row['# features'])
            k = k_row['# features']
            k = int(k)
            
            clf, f_scores = featureSelectParamTuneCLF(features, labels, algo, k)     
        
        else:
            pass
        
        clf.fit(features, labels)
        print clf
 
        
        # validate & collect results for current algo
        precision, recall, f1, f2, accuracy = \
        (test_classifier(clf, my_dataset, features_list, folds=100))
        
        algo_results = [precision, recall, f1, f2, accuracy, clf]
        results.append(algo_results)
        
        print "\r\n* %s F1 is %s, precision is %s, recall is %s. *" % (algo_name, f1, precision, recall)
        print "=" * 10
        
    # create data frame of all algos' results
    results_df = pd.DataFrame(results, index = algo_names, columns = \
                              ['precision', 'recall','f1', 'f2', 'accuracy', 'clf'])
    fi_df = pd.DataFrame(fi_dict, index = features_list[1:])
    
    # output summary
    print "\r\n=== SUMMARY ==="
    print "FUNCTION TYPE: ", kind
    print "FEATURE LIST: ", use_features
    print "DATASET: ", dataset
    print "INCL. NEW FEATURES: ", incl_new
    print "SCORING: ", scoring
    print "NUMBER OF FEATURES: ", num_f
    display = results_df.drop("clf", axis=1)
    print display
    return results_df

###############################################
############## RUN CLASSIFIERS ################
###############################################

### loop through each set of instructions
for instr in instr_list:
    iter_count += 1
    print "\n=== ITERATION %s out of %s " % (iter_count, iterations) + "="*50 + "\n"
    
    ### assign each value in instructions to variables
    kind = instr['kind'] 
    use_features = instr['feature_list']
    dataset = instr['dataset']
    incl_new = instr['new_features']

    ### Select which features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".

    if use_features == 'all':
        # use full feature list
        features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',\
                         'bonus','director_fees', 'restricted_stock_deferred', 'total_stock_value', 'expenses',\
                         'from_poi_to_this_person','loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',\
                         'deferred_income','shared_receipt_with_poi', 'restricted_stock', 'long_term_incentive']

    elif use_features == 'uncorr':
        # use only features with less than .7 correlation with other features
        features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus',
                        'total_stock_value', 'expenses', 'from_poi_to_this_person',
                         'from_messages', 'other', 'deferred_income',
                         'shared_receipt_with_poi', 'long_term_incentive']
    else:
        # use only features with values for > 75 rows
        features_list =['poi', 'salary', 'to_messages', 'total_payments', 'bonus', 'total_stock_value',
                    'shared_receipt_with_poi', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person',
                    'from_this_person_to_poi', 'expenses', 'restricted_stock']


    ### Create new feature(s) (optional)

    if incl_new == 'true':
        # update feature list
        features_list.extend(new_features) 
    else:
        pass
    
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### store number of features
    num_f = len(features[0]) 

    ### Run classifiers
    
    
    if dataset == 'full':
        # cross-validation for use by grid search
        cv = StratifiedShuffleSplit(labels, 100, random_state = 28) 
        
        # send list of algorithms, features & labels 
        # to the appropriate "kind" of function
        results_df = createCLF(kind, features, labels, algos)
    
    else:
        # create training & testing data
        features_train, features_test, labels_train, labels_test = \
                createShuffleTrainTest(features, labels, folds=100)
            
        # cross-validation for use by grid search    
        cv = StratifiedShuffleSplit(labels_train, 100, random_state = 28)
        
        # send list of algorithms, features & labels 
        # to the appropriate "kind" of function
        results_df = createCLF(kind, features_train, labels_train, algos)

    ### identify the best classifier
    ### and save clf + model variables to dict
    
    # record best CLF of this batch based on f1 score
    maxf1 = results_df.f1.idxmax()
    best = results_df.loc[maxf1]
    
    # add time-stamps (for logging later in csv)
    current_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    timestamp = pd.Series([current_time], index=['timestamp'])
    best = best.append(timestamp)
    #new = best.reset_index()
        
    ### create best_dict of clf & scores 
    ### (for logging later in csv)
    best_dict = dict(best)
    
    ### add current iteration's variable values to dict
    best_dict['scoring'] = scoring
    best_dict['# features'] = num_f
    best_dict['mode'] = kind
    best_dict['dataset'] = dataset
    best_dict['feature_set'] = use_features
    best_dict['new_features'] = incl_new
    best_dict['k'] = ''
    
    # get clf from 
    
    # save clf with best f1 so it can be dumped to tester script
    best_clf = best_dict['clf']

    # extract just the 'AlgoName(params)' part
    if kind == 'simple':
        #best_dict['clf'] = best_clf
        pass
    else:
        # retrieve estimator from pipeline
        best_clf_params = best_clf.get_params(deep=True)
        best_clf = best_clf_params['algo']

        if kind.startswith('fs'):
            # get feature selection k value
            k = best_clf_params['select__k']
            best_dict['k'] = k
        else:
            pass

    # now extract just the AlgoName 
    # for better readability in display & csv
    clf_name = getAlgoName(best_clf)
    best_dict['clf'] = clf_name
    
    ### print summary
    print "\r\n=== CONCLUSION ==="
    print "\r\nUsing a %s classifier, with %d features and '%s' scoring, " % (kind, len(features[0]), scoring)
    print "the best algorithm is %s," % maxf1
    print "where f1 is %f, precision is %f, recall is %f\r\n" % (best.f1, best.precision, best.recall)
    print "\n=== ITERATION FINISHED " + "="*50 + "\n"

    ### append best_dict to a csv
    filename = 'ML_proj_results_nov_29_16.csv'
    with open(filename, 'a+') as f:
        fieldnames = best_dict.keys()
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        f.seek(0) # workaround for http://bugs.python.org/issue22651
        content = f.readline()
        # is this a new file, print header
        if content == "":
            dict_writer.writeheader()
        else:
            pass
        dict_writer.writerow(best_dict)
        
########################################################
### If generating a single-model:                    ###
### Dump your best CLF, dataset, and features_list   ###
### so anyone can check your results.                ###
########################################################

### check if there is only 1 set of "instructions" in the 
### instruction list

if iterations == 1:
    try:
        dump_classifier_and_data(best_clf, my_dataset, features_list)
    except NameError:
        pass


print "\r\n====== Finished running script!! ======"
print "\r"