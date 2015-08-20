import math
import random

import numpy as np
import pandas as pd
from sklearn import cross_validation, grid_search, metrics

import utils.unbalanced_dataset as ud


def gen_Xs_ys_split(X, y, test_size=.4):
    """Returns a tuple of (X_train, X_test, y_train, y_test).

    :param loss_threshold: A float less than 0 indicating the minimum loss required to receive a classification of 1
    :param test_size: A float less than 1 indicating the proportion of the available data to reserve for testing
    :rtype: A tuple
    """
    y = y.dropna()
    X = X.ix[y.index].dropna(axis=0, how='all')
    y = y.ix[X.index]
                
    cutoff = int(y.shape[0] * (1 - test_size))
    X_train = X.ix[:cutoff,:] 
    X_test = X.ix[cutoff:,:] 
    y_train = y.ix[:cutoff] 
    y_test = y.ix[cutoff:] 
        
    return (X_train, X_test, y_train, y_test)


def upsample_hits(X, y):    
    """For use with unbalanced classification data sets, duplicate or upsample 1s so that set of 1s and 0s roughly even. Returns (X, y).
    Uses SMOTE algorithm to generate synthetic 1s using k nearest neighbors with 0s data to create more generalizable data.
    SMOTE algorithm from library 'UnbalancedDataset', found here: https://github.com/fmfn/UnbalancedDataset
    :param X: A dataframe of features
    :param y: A series of clasifications
    :rtype: A tuple, (X, y)
    """
    
    # hits = y[y==1]
    # hit_rows = X.ix[hits.index]
    # hit_ratio = (y.sum() * 1.) / y.count()
    
    # for i in range(int(1 / hit_ratio)):
    #     X = X.append(hit_rows)
    #     y = y.append(hits)
    
    # return (X, y)
    ratio = float(np.count_nonzero(y==1)) / float(np.count_nonzero(y==0))
    smote = ud.SMOTE(ratio=(1. / ratio), verbose=False, kind='regular')
    smox, smoy = smote.fit_transform(X.values, y.values)
    X = pd.DataFrame(data=smox, columns=X.columns)
    y = pd.Series(data=smoy)
    return (X, y)



def infer_selected_features(X_train, X_pruned):
    selected_features = []
    for ind in range(X_pruned.shape[1]):
        for c in X_train.columns:
            if (X_train[c] == X_pruned[:,ind]).all():
                selected_features.append(c)
                break
    return selected_features


def cross_validation_metrics(clf, X, y, sample_size=None):
    """
    Runs k-fold (without shuffle) cross validation on data set and returns dictionary with keys for `geometric_mean`
    and `weighted_accuracy` each containing a list of scores.
    
    :param clf: A scikit learn model object, e.g. linear_model.LogisticRegression or ensemble.RandomForestClassifier
    :param X: A dataframe of features
    :param y: A series of clasifications
    :param sample_size: If not None, Float indicating proportion of data set to randomly sample for training model
    :rtype: dict
    """
    cv_metrics = {'geometric_mean': [], 'weighted_accuracy': []}
    kf = cross_validation.KFold(y.shape[0], n_folds=4, shuffle=False)
    for train, test in kf:
        
        X_train_up, y_train_up = upsample_hits(X.ix[train], y.ix[train])
        
        if sample_size:
            length = y_train_up.shape[0]
            inds = sorted(random.sample(range(length), int(length*sample_size)))
            X_train_up = X_train_up.ix[inds]
            y_train_up = y_train_up.ix[inds]
        
        clf.fit(X_train_up, y_train_up)
        y_pred_cv = clf.predict(X.ix[test])
        
        cm = metrics.confusion_matrix(y.ix[test], y_pred_cv)
        tnr = (cm[1,1] * 1.) / (cm[1,0] + cm[1,1])
        tpr = (cm[0,0] * 1.) / (cm[0,0] + cm[0,1])
        
        geo_mean = math.sqrt((tpr * tnr))
        pos_weight = .75
        weighted_acc = pos_weight * tpr + (1-pos_weight) * tnr 
        
        cv_metrics['geometric_mean'].append(geo_mean)
        cv_metrics['weighted_accuracy'].append(weighted_acc)
        
        #fig, axes = plt.subplots(ncols=2, figsize=[18, 3])
        #y.ix[train].plot(ax=axes[0])
        #y.ix[test].plot(ax=axes[1])

    return cv_metrics


def run_grid_search(model, X, y, **params):    
    """Runs exhaustive grid search on model/algorithm and return dataframe with cross validation metrics 
    for each permutation of hyperparameters and returns summarizing dataframe.
    
    :param model: An identifier for a scikit learn model class, e.g. linear_model.LogisticRegression or ensemble.RandomForestClassifier
    :param X: A dataframe of features
    :param y: A series of clasifications
    :**params: Key value pairs to specify the range of hyperparameter values to try, e.g C=[1,2,3] or penalty=('l1','l2')
    :rtype: pandas DataFrame
    """
    grid = []
    for params in grid_search.ParameterGrid(params):
        row = params    
        clf = model(**params)
        cv_metrics = cross_validation_metrics(clf, X, y)   
        row['geometric_mean'] = np.mean(cv_metrics['geometric_mean'])
        row['weighted_accuracy'] = np.mean(cv_metrics['weighted_accuracy'])
        row['metric_mean'] = (row['geometric_mean'] + row['weighted_accuracy']) / 2.
        grid.append(row)
    grid = pd.DataFrame(grid)    
    return grid


def test_model(clf, X_train, X_test, y_train, y_test):
    """Fits model on training data and analyzes performance on test data. Prints generalized empirics to judge.

    :param clf: A scikit learn model object, e.g. linear_model.LogisticRegression or ensemble.RandomForestClassifier
    :param X_train: A pandas dataframe 
    :param X_test: A pandas dataframe
    :param y_train: A pandas dataframe
    :param y_test: A pandas dataframe
    :rtype: None
    """
    X_train_up, y_train_up = upsample_hits(X_train, y_train)
    clf.fit(X_train_up, y_train_up)
    y_pred = clf.predict(X_test)
    print "r-squared:\t{0}".format(clf.score(X_test, y_test))
    print "f1 score:\t{0}".format(metrics.f1_score(y_test, 
                                                   y_pred, 
                                                   labels=[0,1], 
                                                   pos_label=1, 
                                                   average='binary', 
                                                   sample_weight=None))
    print "AUC score:\t{0}".format(metrics.roc_auc_score(y_test, 
                                                         y_pred, 
                                                         average='weighted', 
                                                         sample_weight=None)) 
    print
    print pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), 
                       index=['0 - actual', '1 - actual'], 
                       columns=['0 - predicted', '1 - predicted'])


def plot_decisions_on_train_set(clf, X, y):
    X_up, y_up = upsample_hits(X, y)
    clf.fit(X_up, y_up)
    X.plot(figsize=[18,3], alpha=.1, title="feature scores with model predictions and actual labels")
    pd.Series(clf.predict_proba(X)[:,1], index=X.index).plot(alpha=.8)
    y.plot(alpha=.8)


def plot_learning_curve(clf, X, y):
    grid = {}
    for i in (.1, .25, .5, .75, .9, 1):
        bank = []
        for j in range(5):    
            cv_metrics = cross_validation_metrics(clf, X, y, sample_size=i)
            avg_score = (np.mean(cv_metrics['geometric_mean']) + np.mean(cv_metrics['weighted_accuracy'])) / 2.
            bank.append(avg_score)
        grid[i] = np.mean(bank)
    grid = pd.Series(grid, name='cv_score')
    grid.index.name = 'proportion of total data set used'
    grid.plot(title="learning curve (cv scores)", figsize=[9,3])