from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np


def CrossValidation (X,y,groups):
    fold = 2
    scorces = dict()

    SVM = svm.LinearSVC()
    RF = RandomForestClassifier(max_depth=None, n_estimators=200)
    LR = LogisticRegression()

    est = {'SVM': SVM, 'RandomForest': RF, 'LogisticRegression': LR}

    loo_c = LeaveOneOut()
    loo = loo_c.split(X,y)

    group_kfold_c = GroupKFold(n_splits=fold)
    group_kfold = group_kfold_c.split(X,y,groups)

    skf_c = StratifiedKFold(n_splits=fold)
    skf = skf_c.split(X,y,groups)

    CVM={'loo':loo,'GroupKFold':group_kfold,'StratifiedKFold':skf}

    for estimator in est.keys():
        scorces[estimator] = dict()
        for cv in CVM.keys():
            scorces[estimator][cv] = np.average(cross_val_score(estimator=est[estimator],X=X,y=y,
                                                                cv=CVM[cv],scoring='roc_auc'))

    df = pd.DataFrame(scorces.items)
    df.transpose()






