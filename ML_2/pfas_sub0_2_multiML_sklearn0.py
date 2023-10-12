# multi label classification
# binary relevant/ classifier chains/ label powerset/adjust ml (knn,rf -- scikit learn)

import scipy
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import numpy as np

# multi-label packages
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score        # accuracy
from sklearn.metrics import mean_squared_error    # Mean Squared Error (MSE)
import sklearn.metrics as metrics                 # f-i score

# XGBoost/CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
# https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f

from sklearn.datasets import make_multilabel_classification


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # add the parameters needed to change
    parser.add_argument("--which_model", type=str, default='CatBoost_chain',help='BiRel1, ClassifierChain2, LabelPowerset3，XGBoost,CatBoostClassifier,ML-KNN,CatBoost_bire,CatBoost_chain, ')
    # parser.add_argument("--batch_size", type=int, default=32,help='batch size')   # default 16
    # parser.add_argument("--output_folderName", type=str, default='ML_Result',help='')
    return parser

# X_try, y_try = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)

def main( ):

    # parameters
    # parser = config_parser()
    # args = parser.parse_args()
    # model_name = args.which_model
    # print('model_name: ',model_name)

    # Load Data
    df = pd.read_csv("GW_MAIN_PFAS_14_subset_0_super.csv")
    df = df.drop(columns=['gm_gis_dwr_region'])     # 'latitude', 'longitude',

    # 将数据的feature和label分开，同时将数据分成训练集合测试集
    X = df.iloc[:, 0:113].values     # 114 # 111
    #X.toarray()  # convert to dense numpy array
    y = df.iloc[:, 113:].values     # .values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # y_train = y_train.astype(np.float64)
    # y_test = y_test.astype(np.float64)


    ###
    model_name = 'XGBoost_chain'
    print(model_name)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=2)    # random_state=2 结果可以被复现
    print(len(xtest))
    # ClassifierChain
    classifier = ClassifierChain(XGBClassifier())
    clf = Pipeline([('classify', classifier)])
    print(clf)

    clf.fit(xtrain, ytrain)
    print(clf.score(xtrain, ytrain))
    print('test score', clf.score(xtest, ytest))

    yhat = clf.predict(xtest)
    accuracy = accuracy_score(ytest,yhat)
    print('accuracy: ',accuracy)

    # classifier chain
    auc_y1 = roc_auc_score(ytest[:, 0], yhat[:, 0].toarray())
    auc_y2 = roc_auc_score(ytest[:, 1], yhat[:, 1].toarray())
    auc_y3 = roc_auc_score(ytest[:, 2], yhat[:, 2].toarray())
    auc_y4 = roc_auc_score(ytest[:, 3], yhat[:, 3].toarray())
    auc_y5 = roc_auc_score(ytest[:, 4], yhat[:, 4].toarray())
    auc_y6 = roc_auc_score(ytest[:, 5], yhat[:, 5].toarray())
    auc_y7 = roc_auc_score(ytest[:, 6], yhat[:, 6].toarray())
    auc_y8 = roc_auc_score(ytest[:, 7], yhat[:, 7].toarray())
    auc_y9 = roc_auc_score(ytest[:, 8], yhat[:, 8].toarray())
    auc_y10 = roc_auc_score(ytest[:, 9], yhat[:, 9].toarray())
    auc_y11 = roc_auc_score(ytest[:, 10], yhat[:, 10].toarray())
    auc_y12 = roc_auc_score(ytest[:, 11], yhat[:, 11].toarray())
    auc_y13 = roc_auc_score(ytest[:, 12], yhat[:, 12].toarray())
    auc_y14 = roc_auc_score(ytest[:, 13], yhat[:, 13].toarray())
    auc_total = np.mean([auc_y1,auc_y2,auc_y3,auc_y4,auc_y5,auc_y6,auc_y7,auc_y8,auc_y9,auc_y10,auc_y11,auc_y12,auc_y13,auc_y14])
    print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f, y5: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4, auc_y5))
    print("ROC AUC total ave: ", auc_total)
    # xgboost https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f




if __name__ == "__main__":
    main( )


print('\n--------------------------------------------------------------\n')

###
model_name = 'XGBoost_bire'
'''
print(model_name)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=2)    # random_state=2 结果可以被复现
print(len(xtest))
# binary rel
classifier = MultiOutputClassifier(XGBClassifier(use_label_encoder=False))
clf = Pipeline([('classify', classifier)])
print(clf)

clf.fit(xtrain, ytrain)
print('train score',clf.score(xtrain, ytrain))
print('test score', clf.score(xtest, ytest))

yhat = clf.predict(xtest)
accuracy = accuracy_score(ytest,yhat)
print('accuracy: ',accuracy)

auc_y1 = roc_auc_score(ytest[:, 0], yhat[:, 0])
auc_y2 = roc_auc_score(ytest[:, 1], yhat[:, 1])
auc_y3 = roc_auc_score(ytest[:, 2], yhat[:, 2])
auc_y4 = roc_auc_score(ytest[:, 3], yhat[:, 3])
auc_y5 = roc_auc_score(ytest[:, 4], yhat[:, 4])
auc_y6 = roc_auc_score(ytest[:, 5], yhat[:, 5])
auc_y7 = roc_auc_score(ytest[:, 6], yhat[:, 6])
auc_y8 = roc_auc_score(ytest[:, 7], yhat[:, 7])
auc_y9 = roc_auc_score(ytest[:, 8], yhat[:, 8])
auc_y10 = roc_auc_score(ytest[:, 9], yhat[:, 9])
auc_y11 = roc_auc_score(ytest[:, 10], yhat[:, 10])
auc_y12 = roc_auc_score(ytest[:, 11], yhat[:, 11])
auc_y13 = roc_auc_score(ytest[:, 12], yhat[:, 12])
auc_y14 = roc_auc_score(ytest[:, 13], yhat[:, 13])
auc_total = np.mean([auc_y1,auc_y2,auc_y3,auc_y4,auc_y5,auc_y6,auc_y7,auc_y8,auc_y9,auc_y10,auc_y11,auc_y12,auc_y13,auc_y14])
print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f, y5: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4, auc_y5))
print("ROC AUC total ave: ", auc_total)

cm_y1 = confusion_matrix(ytest[:, 0], yhat[:, 0])
cm_y2 = confusion_matrix(ytest[:, 1], yhat[:, 1])
cm_y3 = confusion_matrix(ytest[:, 2], yhat[:, 2])
cm_y4 = confusion_matrix(ytest[:, 3], yhat[:, 3])
cm_y5 = confusion_matrix(ytest[:, 4], yhat[:, 4])
print(cm_y1,cm_y2,cm_y3,cm_y4,cm_y5)

cr_y1 = classification_report(ytest[:, 0], yhat[:, 0])
cr_y2 = classification_report(ytest[:, 1], yhat[:, 1])
cr_y3 = classification_report(ytest[:, 2], yhat[:, 2])
cr_y4 = classification_report(ytest[:, 3], yhat[:, 3])
cr_y5 = classification_report(ytest[:, 4], yhat[:, 4])
print(cr_y1,cr_y2,cr_y3,cr_y4,cr_y5)
'''

# https://www.zhihu.com/question/35486862


