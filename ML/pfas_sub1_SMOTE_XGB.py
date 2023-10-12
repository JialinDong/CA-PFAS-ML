# multi label classification  --  DT & RandomForestClassifier

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

import imblearn       # 使用不平衡学习 Python 库提供的实现
print(imblearn.__version__)
from collections import Counter

import xgboost
from xgboost import XGBClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from numpy import where
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN   # 自适应合成采样 (ADASYN)

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def create_dataset(n_sample=1000):
    '''
    Create a unevenly distributed sample data set multilabel
    classification using make_classification function
    args
    nsample: int, Number of sample to be created
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2,weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y

def get_tail_label(df,j):
    """
    Give tail label colums of the given target dataframe
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    return
    tail_label: list, a list containing column name of all the tail label
    """
    # calculate for the Imbalance ratio per label / Mean Imbalance ratio
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    if j == 1:
        for i in range(n):
            if irpl[i] > mir:              # old
                tail_label.append(columns[i])
    else:
        for i in range(n):
            if irpl[i] < mir:              # change > to <
                tail_label.append(columns[i])
    return tail_label

def get_index(df,j):
    """
    give the index of all tail_label rows
    args
    df: pandas.DataFrame, target label df from which index for tail label has to identified
    return
    index: list, a list containing index number of all the tail label
    """
    tail_labels = get_tail_label(df,j)
    print('tail_labels', tail_labels)
    index = set()
    i = 0
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 0].index)   # change 1 to 0 in pfas dataset
        if i == 0:
            index = index.union(sub_index)     # change union to intersection
            i = i+1
        else:
            index = index.intersection(sub_index)
    return list(index)

def get_minority_instace(X, y, j):
    """
    Give minority dataframe containing all the tail labels
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y, j)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub

def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    args
    X: np.array, array whose nearest neighbor has to find
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=14, metric='euclidean', algorithm='kd_tree').fit(X)   # change the neighbors based on label?
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X_raw, y_raw, X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X_raw, new_X], axis=0)
    target = pd.concat([y_raw, target], axis=0)
    return new_X, target

def oversampling(X_raw, y_raw, X, y):
    new_X = pd.concat([X_raw, X], axis=0)
    target = pd.concat([y_raw, y], axis=0)
    return new_X, target

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # add the parameters needed to change
    parser.add_argument("--which_data", type=str, default='pfas_Pseudo_labeling_XBG_diffTEST',help='pfas_Pseudo_binary_XBG,pfas,pfas_MLSMOTE，example')
    # pfas_Pseudo_multi-XBG  pfas_Pseudo_binary_XBG pfas_Pseudo_labeling_XBG pfas_Pseudo_labeling_XBG_diffTEST
    return parser

def main( ):
    # parameters
    parser = config_parser()
    args = parser.parse_args()

    file_name = args.which_data
    print('file_name',file_name)

    if file_name == 'pfas_MLSMOTE':
        print('pfas dataset -- pfas_MLSMOTE')
        # Load Data
        # df = pd.read_csv("GW_MAIN_PFAS_14_subset_0_super.csv")  # 0
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_super.csv")       # 1
        # df = pd.read_csv("GW_MAIN_PFAS_14_subset_2_super.csv")       # 2
        # df = pd.read_csv("GW_MAIN_PFAS_14_subset_3_super.csv")       # 3

        df = df.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',
        # 将数据的feature和label分开，同时将数据分成训练集合测试集
        X = df.iloc[:, 0:113]
        y = df.iloc[:, 113:]  # .values

        # 1 SMOTE Oversampling for Multi-Class Classification
        print('raw dataset')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0]/y.iloc[:, i].count()
            print( i, list(y.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X, y, 0)  # Getting minority instance of that dataframe
        X_res, y_res = MLSMOTE(X, y, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 1')
        for i in range(y_res.shape[1]):
            counter = Counter(y_res.iloc[:, i])
            rate = counter[0]/y_res.iloc[:, i].count()
            print( i, list(y_res.columns.values)[i], ': ', counter, 'rate:' , rate)
        ## repeat the process until the dataset is balanced
        X_sub, y_sub = get_minority_instace(X_res, y_res, 0)  # Getting minority instance of that dataframe
        X_res_1, y_res_1 = MLSMOTE(X_res, y_res, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 2')
        for i in range(y_res_1.shape[1]):
            counter = Counter(y_res_1.iloc[:, i])
            rate = counter[0]/y_res_1.iloc[:, i].count()
            print( i, list(y_res_1.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X_res_1, y_res_1, 0)  # Getting minority instance of that dataframe
        X_res_2, y_res_2 = MLSMOTE(X_res_1, y_res_1, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 3')
        for i in range(y_res_2.shape[1]):
            counter = Counter(y_res_2.iloc[:, i])
            rate = counter[0] / y_res_2.iloc[:, i].count()
            print(i, list(y_res_2.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_2, y_res_2, 1)  # Getting minority instance of that dataframe
        X_res_3, y_res_3 = MLSMOTE(X_res_2, y_res_2, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 4 -- change > ')
        for i in range(y_res_3.shape[1]):
            counter = Counter(y_res_3.iloc[:, i])
            rate = counter[0] / y_res_3.iloc[:, i].count()
            print(i, list(y_res_3.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_3, y_res_3, 1)  # Getting minority instance of that dataframe
        X_res_4, y_res_4 = MLSMOTE(X_res_3, y_res_3, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 5 -- change > ')
        for i in range(y_res_4.shape[1]):
            counter = Counter(y_res_4.iloc[:, i])
            rate = counter[0] / y_res_4.iloc[:, i].count()
            print(i, list(y_res_4.columns.values)[i], ': ', counter, 'rate:', rate)

        # oversampling - duplicated the get_minority_instace
        X_sub, y_sub = get_minority_instace(X_res_4, y_res_4, 0)  # Getting minority instance of that dataframe
        X_res_5, y_res_5 = oversampling(X_res_4, y_res_4, X_sub, y_sub)  #
        # summarize class distribution
        print('oversampling dataset 6 -- < duplicated the get_minority_instace ')
        for i in range(y_res_5.shape[1]):
            counter = Counter(y_res_5.iloc[:, i])
            rate = counter[0] / y_res_5.iloc[:, i].count()
            print(i, list(y_res_5.columns.values)[i], ': ', counter, 'rate:', rate)

        # oversampling - duplicated the get_minority_instace
        '''
        X_sub, y_sub = get_minority_instace(X_res_5, y_res_5, 0)  # Getting minority instance of that dataframe
        X_res_6, y_res_6 = oversampling(X_res_5, y_res_5, X_sub, y_sub)  #
        # summarize class distribution
        print('oversampling dataset 7 -- < duplicated the get_minority_instace ')
        for i in range(y_res_6.shape[1]):
            counter = Counter(y_res_6.iloc[:, i])
            rate = counter[0] / y_res_6.iloc[:, i].count()
            print(i, list(y_res_6.columns.values)[i], ': ', counter, 'rate:', rate)
        '''

        print('finished data oversampling')

        # 2 XGBoost prediction accuracy
        print('2 XGBoost prediction accuracy')
        model_name = 'XGBoost_chain'
        print(model_name)
        xtrain, xtest, ytrain, ytest = train_test_split(X_res_5.values, y_res_5.values, train_size=0.8, random_state=2)  # random_state=2 结果可以被复现
        print(len(xtest))
        # ClassifierChain
        classifier = ClassifierChain(XGBClassifier())
        clf = Pipeline([('classify', classifier)])
        print(clf)

        clf.fit(xtrain, ytrain)
        print(clf.score(xtrain, ytrain))
        print('test score', clf.score(xtest, ytest))

        yhat = clf.predict(xtest)
        accuracy = accuracy_score(ytest, yhat)
        print('accuracy: ', accuracy)

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
        auc_total = np.mean([auc_y1, auc_y2, auc_y3, auc_y4, auc_y5, auc_y6, auc_y7, auc_y8, auc_y9, auc_y10, auc_y11, auc_y12, auc_y13,auc_y14])
        print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f, y5: %.4f, y6: %.4f, y7: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4, auc_y5,auc_y6, auc_y7))
        print("ROC AUC y8: %.4f, y9: %.4f, y10: %.4f, y11: %.4f, y12: %.4f, y13: %.4f, y14: %.4f" % (auc_y8, auc_y9, auc_y10, auc_y11, auc_y12, auc_y13, auc_y14))
        print("ROC AUC total ave: ", auc_total)

        print('finished XGBoost_chain prediction')

        # 3 Semi-surperviose ML 伪标签

    # binary models
    elif file_name == 'pfas_Pseudo_binary_XBG':
        print('pfas_Pseudo_binary_XBG')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_super.csv")       # 0
        df = df.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',

        # 找出含有空值的行
        df_semi = pd.read_csv("GW_MAIN_PFAS_14_subset_1_all.csv")       # 0
        df_semi = df_semi.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',
        df_semi = df_semi[df_semi.isnull().T.any()]

        for i in range(len(df.iloc[:, 113:].columns)):
            col_name = df.iloc[:, 113:].columns[i]
            print(i, col_name)
            label_start_no = 113+i
            i = i + 1
            label_end_no = 113 + i
            X = df.iloc[:, 0:label_start_no]
            y = df.iloc[:, label_start_no:label_end_no]       # .values
            counter = Counter(np.array(y[col_name]))
            print('Count for raw data',counter)
            # oversampling -- later
            # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
            print('ADASYN')
            oversample = ADASYN()
            X, y = oversample.fit_resample(X, y)  # Counter({0: 19121, 1: 19121})
            counter = Counter(np.array(y[col_name]))
            print('Count after oversample',counter)
            # split to train, test
            x_train, x_test_real, y_train, y_test_real = train_test_split(X.values, y.values, test_size=0.2, random_state=0)  # Training Data: real data # test data: Pseudo Labeling

            x_test = df_semi.iloc[:, 0:label_start_no].reset_index(drop=True)    # x_train -- semi-ml data

            TH = 0.98        # 某一筆測試資料屬於標籤0的機率
            np.random.seed(1)

            # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)
            model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False
            y_pred_real = model.predict(x_test_real)
            print('Before PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            print('Before PseudoLabeling', x_train.shape, y_train.shape)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)

            # 使用Pseudo Labeling增加模型效能
            # x_pseudo = [x_test[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            # y_pseudo = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            index_semi = []
            for index in range(len(x_test)):
                if (y_prob[index][0] >= TH or y_prob[index][1] >= TH):
                    index_semi.append(index)
            x_pseudo = x_test[x_test.index.isin(index_semi)].reset_index(drop=True)
            y_pred = pd.DataFrame(y_pred)
            y_pseudo = y_pred[y_pred.index.isin(index_semi)].reset_index(drop=True)
            # y_right0 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH and y_test[index] == 0)]
            # y_right1 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][1] >= TH and y_test[index] == 1)]
            x_train = np.concatenate((x_train, x_pseudo), axis=0)
            y_train = np.concatenate((y_train, y_pseudo), axis=0)
            # check dataset balance
            y_train = pd.DataFrame(y_train)
            counter = Counter(y_train[0])
            print('Count for the PseudoLabeling dataset',counter)
            # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)   # Training Data + Pseudo Labeling
            model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False   xgb.
            y_pred_real = model.predict(x_test_real)
            print('after PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            print('after PseudoLabeling', x_train.shape, y_train.shape)

    # latest version -- final result
    # get the Pseudo labels and train the ml again (split the train/test, then balance, then used the train_p=label to train ml)
    elif file_name == 'pfas_Pseudo_labeling_XBG':
        print('pfas_Pseudo_labeling_XBG')
        # Pseudo labels

        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_super.csv")       # 1
        df = df.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',

        # 找出含有空值的行
        df_semi = pd.read_csv("GW_MAIN_PFAS_14_subset_1_all.csv")       # 1
        df_semi = df_semi.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',
        df_semi = df_semi[df_semi.isnull().T.any()]
        df_semi = df_semi.reset_index(drop=True)

        # get the semi-ml data to train (add new label back to the old dataset and save)
        # create a new dataframe for semi-ml (features + p-label)
        semi_dataframe_p_label = pd.DataFrame(df_semi['date'])
        semi_dataframe_col = list(df.iloc[:, 113:].columns.values)
        semi_dataframe_p_label = pd.concat([semi_dataframe_p_label,pd.DataFrame(columns=semi_dataframe_col)])
        del semi_dataframe_p_label['date']
        # loop over all the labels
        for i in range(len(df.iloc[:, 113:].columns)):
            col_name = df.iloc[:, 113:].columns[i]
            print(i, col_name)
            label_start_no = 113+i
            i = i + 1
            label_end_no = 113 + i
            X = df.iloc[:, 0:label_start_no]
            y = df.iloc[:, label_start_no:label_end_no]       # .values
            counter = Counter(np.array(y[col_name]))
            print('Count for raw data',counter)
            # oversampling -- later
            # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
            print('ADASYN')
            oversample = ADASYN()
            X, y = oversample.fit_resample(X, y)  # Counter({0: 19121, 1: 19121})
            counter = Counter(np.array(y[col_name]))
            print('Count after oversample',counter)
            # split to train, test
            x_train, x_test_real, y_train, y_test_real = train_test_split(X.values, y.values, test_size=0.2, random_state=0)  # Training Data: real data # test data: Pseudo Labeling

            x_test = df_semi.iloc[:, 0:label_start_no]     # .reset_index(replace=True)    # x_train -- semi-ml data

            TH = 0.98        # 某一筆測試資料屬於標籤0的機率
            np.random.seed(1)

            # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)
            model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False
            y_pred_real = model.predict(x_test_real)
            print('Before PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            print('Before PseudoLabeling', x_train.shape, y_train.shape)
            y_train = pd.DataFrame(y_train)
            counter = Counter(y_train[0])
            print('Count for the PseudoLabeling dataset',counter)

            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)

            # 使用Pseudo Labeling增加模型效能
            # x_pseudo = [x_test[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            # y_pseudo = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            index_semi = []
            for index in range(len(x_test)):
                if (y_prob[index][0] >= TH or y_prob[index][1] >= TH):
                    index_semi.append(index)
            if i == 1:
                index_semi_all = index_semi
            else:
                index_semi_all = list(set(index_semi_all).intersection(set(index_semi)))
            print('length index_semi_all',len(index_semi_all))
            x_pseudo = x_test[x_test.index.isin(index_semi)]      # .reset_index(drop=True)
            y_pred = pd.DataFrame(y_pred)
            y_pseudo = y_pred[y_pred.index.isin(index_semi)]      # .reset_index(drop=True)
            ## y_right0 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH and y_test[index] == 0)]
            ## y_right1 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][1] >= TH and y_test[index] == 1)]
            # x_train = np.concatenate((x_train, x_pseudo), axis=0)  #纵向拼接
            # y_train = np.concatenate((y_train, y_pseudo), axis=0)
            ### train teh concate data again to see whether the acc is increased
            # check dataset balance
            # y_train = pd.DataFrame(y_train)
            # counter = Counter(y_train[0])
            # print('Count for the PseudoLabeling dataset',counter)
            # # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)   # Training Data + Pseudo Labeling
            # model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False   xgb.
            # y_pred_real = model.predict(x_test_real)
            # print('after PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            # print('after PseudoLabeling', x_train.shape, y_train.shape)
            # save the p label in semi_dataframe_p_label
            for index_semi_i in index_semi_all:
                semi_dataframe_p_label.at[index_semi_i, col_name] = y_pseudo.at[index_semi_i, 0]

        # drop the rows with any nans -- semi_dataframe_p_label
        # semi_dataframe_p_label = semi_dataframe_p_label.dropna(how='any')  # or
        semi_dataframe_p_label = semi_dataframe_p_label[semi_dataframe_p_label.index.isin(index_semi_all)]
        semi_dataframe_real_label = df_semi[df_semi.index.isin(index_semi_all)]
        semi_dataframe_real_label = semi_dataframe_real_label.iloc[:, 113:]
        # replace/filling na from semi_dataframe_p_label to semi_dataframe_real_label (将_x中的空值用_y中的非空值进行填充)
        semi_dataframe_final_label = semi_dataframe_real_label.fillna(semi_dataframe_p_label,inplace=False)
        # features
        semi_dataframe_features = df_semi.iloc[:, 0:113]
        semi_dataframe_features = semi_dataframe_features[semi_dataframe_features.index.isin(index_semi_all)]
        df_semi_data_only = pd.concat([semi_dataframe_features,semi_dataframe_final_label],axis=1)  # 横向拼接
        df_semi_data_only.to_csv('GW_MAIN_PFAS_14_subset_1_SemiONLY.csv', index=False)  # save file
        #df_semi_data = df_semi[df_semi.index.isin(index_semi_all)]     # .reset_index(drop=True)
        #df_semi_data = pd.DataFrame(df_semi_data)
        # df_semi_add = np.concatenate((df, df_semi_data_add), axis=0)     # 纵向
        # df_semi_add = pd.concat([df,df_semi_data_only],axis=0,ignore_index=True)   # 纵向
        # df_semi_add.to_csv('GW_MAIN_PFAS_14_subset_1_SemiAll.csv', index=False)  # save file


        df_semi_data_only = pd.read_csv("GW_MAIN_PFAS_14_subset_1_SemiONLY.csv")
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_super.csv")
        df = df.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',

        ### oversample & train the multi-label XGB-chain with all data (include the P-label)
        model_name = 'XGBoost_chain'
        print(model_name)

        # 将数据的feature和label分开，同时将数据分成训练集合测试集
        X = df.iloc[:, 0:113]      # 114 # 111
        # X.toarray()  # convert to dense numpy array
        y = df.iloc[:, 113:]       # .values
        # get the train/test data -- # get the test data
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=2)  # random_state=2 结果可以被复现
        print('len(test)', len(xtest))
        # add the p-label data to the train data
        train_all = pd.concat([xtrain, ytrain], axis=1)  # 横向拼接
        train_all = pd.concat([train_all, df_semi_data_only], axis=0)  # 纵拼接

        # 将数据的feature和label分开
        X = train_all.iloc[:, 0:113]      # 114 # 111
        # X.toarray()  # convert to dense numpy array
        y = train_all.iloc[:, 113:]       # .values

        # 1 SMOTE Oversampling for Multi-Class Classification
        print('raw dataset')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0]/y.iloc[:, i].count()
            print(i, list(y.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X, y, 0)  # Getting minority instance of that dataframe
        X, y = MLSMOTE(X, y, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 1')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0]/y.iloc[:, i].count()
            print( i, list(y.columns.values)[i], ': ', counter, 'rate:' , rate)
        ## repeat the process until the dataset is balanced
        '''
        X_sub, y_sub = get_minority_instace(X_res, y_res, 0)  # Getting minority instance of that dataframe
        X_res_1, y_res_1 = MLSMOTE(X_res, y_res, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 2')
        for i in range(y_res_1.shape[1]):
            counter = Counter(y_res_1.iloc[:, i])
            rate = counter[0]/y_res_1.iloc[:, i].count()
            print( i, list(y_res_1.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X_res_1, y_res_1, 0)  # Getting minority instance of that dataframe
        X_res_2, y_res_2 = MLSMOTE(X_res_1, y_res_1, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 3')
        for i in range(y_res_2.shape[1]):
            counter = Counter(y_res_2.iloc[:, i])
            rate = counter[0] / y_res_2.iloc[:, i].count()
            print(i, list(y_res_2.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_2, y_res_2, 1)  # Getting minority instance of that dataframe
        X_res_3, y_res_3 = MLSMOTE(X_res_2, y_res_2, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 4 -- change > ')
        for i in range(y_res_3.shape[1]):
            counter = Counter(y_res_3.iloc[:, i])
            rate = counter[0] / y_res_3.iloc[:, i].count()
            print(i, list(y_res_3.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_3, y_res_3, 1)  # Getting minority instance of that dataframe
        X_res_4, y_res_4 = MLSMOTE(X_res_3, y_res_3, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 5 -- change > ')
        for i in range(y_res_4.shape[1]):
            counter = Counter(y_res_4.iloc[:, i])
            rate = counter[0] / y_res_4.iloc[:, i].count()
            print(i, list(y_res_4.columns.values)[i], ': ', counter, 'rate:', rate)

        # oversampling - duplicated the get_minority_instace
        X_sub, y_sub = get_minority_instace(X_res_4, y_res_4, 0)  # Getting minority instance of that dataframe
        X, y = oversampling(X_res_4, y_res_4, X_sub, y_sub)  #
        # summarize class distribution
        print('oversampling dataset 6 -- < duplicated the get_minority_instace ')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0] / y.iloc[:, i].count()
            print(i, list(y.columns.values)[i], ': ', counter, 'rate:', rate)
        '''

        # 2 train the model
        # ClassifierChain
        classifier = ClassifierChain(XGBClassifier())
        clf = Pipeline([('classify', classifier)])
        print(clf)

        clf.fit(xtrain, ytrain)
        print(clf.score(xtrain, ytrain))
        print('test score', clf.score(xtest, ytest))

        yhat = clf.predict(xtest)
        accuracy = accuracy_score(ytest, yhat)
        print('accuracy: ', accuracy)
        print('precision_score', precision_score(y_true=ytest, y_pred=yhat, average='samples'))
        print('recall_score',recall_score(y_true=ytest, y_pred=yhat, average='samples'))
        print('f1_score', f1_score(ytest, yhat, average='samples'))
        # Hamming Loss -- 衡量的是所有样本中，预测错的标签数在整个标签标签数中的占比. 其值越小表示模型的表现结果越好
        print('hamming_loss',hamming_loss(ytest, yhat))

        ytest = ytest.values
        # classifier chain
        auc_y1 = roc_auc_score(ytest[:, 0], yhat[:, 0].toarray())
        auc_y2 = roc_auc_score(ytest[:, 1], yhat[:, 1].toarray())
        auc_y3 = roc_auc_score(ytest[:, 2], yhat[:, 2].toarray())
        auc_y4 = roc_auc_score(ytest[:, 3], yhat[:, 3].toarray())
        auc_total = np.mean([auc_y1, auc_y2, auc_y3, auc_y4])
        print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4))
        print("ROC AUC total ave: ", auc_total)

        cm_y1 = confusion_matrix(ytest[:, 0], yhat[:, 0].toarray())
        cm_y2 = confusion_matrix(ytest[:, 1], yhat[:, 1].toarray())
        cm_y3 = confusion_matrix(ytest[:, 2], yhat[:, 2].toarray())
        cm_y4 = confusion_matrix(ytest[:, 3], yhat[:, 3].toarray())
        print(cm_y1, cm_y2, cm_y3, cm_y4)

        cr_y1 = classification_report(ytest[:, 0], yhat[:, 0].toarray())
        cr_y2 = classification_report(ytest[:, 1], yhat[:, 1].toarray())
        cr_y3 = classification_report(ytest[:, 2], yhat[:, 2].toarray())
        cr_y4 = classification_report(ytest[:, 3], yhat[:, 3].toarray())
        print(cr_y1, cr_y2, cr_y3, cr_y4)

        # xgboost https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f
        # XGBoost https://www.kaggle.com/code/para24/ovr-vs-multioutput-vs-classifier-chaining/notebook

    # balance then split to train, test
    elif file_name == 'pfas_Pseudo_labeling_XBG_diffTEST':
        print('pfas_Pseudo_labeling_XBG_diffTEST')
        # Pseudo labels

        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_super.csv")       # 1
        df = df.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',

        # 找出含有空值的行
        df_semi = pd.read_csv("GW_MAIN_PFAS_14_subset_1_all.csv")       # 1
        df_semi = df_semi.drop(columns=['gm_gis_dwr_region'])  # 'latitude', 'longitude',
        df_semi = df_semi[df_semi.isnull().T.any()]
        df_semi = df_semi.reset_index(drop=True)

        # get the semi-ml data to train (add new label back to the old dataset and save)
        # create a new dataframe for semi-ml (features + p-label)
        semi_dataframe_p_label = pd.DataFrame(df_semi['date'])
        semi_dataframe_col = list(df.iloc[:, 113:].columns.values)
        semi_dataframe_p_label = pd.concat([semi_dataframe_p_label,pd.DataFrame(columns=semi_dataframe_col)])
        del semi_dataframe_p_label['date']
        # loop over all the labels
        for i in range(len(df.iloc[:, 113:].columns)):
            col_name = df.iloc[:, 113:].columns[i]
            print(i, col_name)
            label_start_no = 113+i
            i = i + 1
            label_end_no = 113 + i
            X = df.iloc[:, 0:label_start_no]
            y = df.iloc[:, label_start_no:label_end_no]       # .values
            counter = Counter(np.array(y[col_name]))
            print('Count for raw data',counter)
            # oversampling -- later
            # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
            print('ADASYN')
            oversample = ADASYN()
            X, y = oversample.fit_resample(X, y)  # Counter({0: 19121, 1: 19121})
            counter = Counter(np.array(y[col_name]))
            print('Count after oversample',counter)
            # split to train, test
            x_train, x_test_real, y_train, y_test_real = train_test_split(X.values, y.values, test_size=0.2, random_state=0)  # Training Data: real data # test data: Pseudo Labeling

            x_test = df_semi.iloc[:, 0:label_start_no]     # .reset_index(replace=True)    # x_train -- semi-ml data

            TH = 0.98        # 某一筆測試資料屬於標籤0的機率
            np.random.seed(1)

            # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)
            model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False
            y_pred_real = model.predict(x_test_real)
            print('Before PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            print('Before PseudoLabeling', x_train.shape, y_train.shape)
            y_train = pd.DataFrame(y_train)
            counter = Counter(y_train[0])
            print('Count for the PseudoLabeling dataset',counter)

            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)

            # 使用Pseudo Labeling增加模型效能
            # x_pseudo = [x_test[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            # y_pseudo = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH or y_prob[index][1] >= TH)]
            index_semi = []
            for index in range(len(x_test)):
                if (y_prob[index][0] >= TH or y_prob[index][1] >= TH):
                    index_semi.append(index)
            if i == 1:
                index_semi_all = index_semi
            else:
                index_semi_all = list(set(index_semi_all).intersection(set(index_semi)))
            print('length index_semi_all',len(index_semi_all))
            x_pseudo = x_test[x_test.index.isin(index_semi)]      # .reset_index(drop=True)
            y_pred = pd.DataFrame(y_pred)
            y_pseudo = y_pred[y_pred.index.isin(index_semi)]      # .reset_index(drop=True)
            ## y_right0 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][0] >= TH and y_test[index] == 0)]
            ## y_right1 = [y_pred[index] for index in range(len(x_test)) if (y_prob[index][1] >= TH and y_test[index] == 1)]
            # x_train = np.concatenate((x_train, x_pseudo), axis=0)  #纵向拼接
            # y_train = np.concatenate((y_train, y_pseudo), axis=0)
            ### train teh concate data again to see whether the acc is increased
            # check dataset balance
            # y_train = pd.DataFrame(y_train)
            # counter = Counter(y_train[0])
            # print('Count for the PseudoLabeling dataset',counter)
            # # model = RandomForestClassifier(random_state=0).fit(x_train, y_train)   # Training Data + Pseudo Labeling
            # model = XGBClassifier(use_label_encoder=False).fit(x_train, y_train)   # use_label_encoder=False   xgb.
            # y_pred_real = model.predict(x_test_real)
            # print('after PseudoLabeling F1 - Score:', f1_score(y_test_real, y_pred_real))
            # print('after PseudoLabeling', x_train.shape, y_train.shape)
            # save the p label in semi_dataframe_p_label
            for index_semi_i in index_semi_all:
                semi_dataframe_p_label.at[index_semi_i, col_name] = y_pseudo.at[index_semi_i, 0]

        # drop the rows with any nans -- semi_dataframe_p_label
        # semi_dataframe_p_label = semi_dataframe_p_label.dropna(how='any')  # or
        semi_dataframe_p_label = semi_dataframe_p_label[semi_dataframe_p_label.index.isin(index_semi_all)]
        semi_dataframe_real_label = df_semi[df_semi.index.isin(index_semi_all)]
        semi_dataframe_real_label = semi_dataframe_real_label.iloc[:, 113:]
        # replace/filling na from semi_dataframe_p_label to semi_dataframe_real_label (将_x中的空值用_y中的非空值进行填充)
        semi_dataframe_final_label = semi_dataframe_real_label.fillna(semi_dataframe_p_label,inplace=False)
        # features
        semi_dataframe_features = df_semi.iloc[:, 0:113]
        semi_dataframe_features = semi_dataframe_features[semi_dataframe_features.index.isin(index_semi_all)]
        df_semi_data_only = pd.concat([semi_dataframe_features,semi_dataframe_final_label],axis=1)  # 横向拼接
        # df_semi_data_only.to_csv('GW_MAIN_PFAS_14_subset_1_SemiONLY.csv', index=False)  # save file
        #df_semi_data = df_semi[df_semi.index.isin(index_semi_all)]     # .reset_index(drop=True)
        #df_semi_data = pd.DataFrame(df_semi_data)
        df_semi_add = pd.concat([df,df_semi_data_only],axis=0,ignore_index=True)   # 纵向
        df_semi_add.to_csv('GW_MAIN_PFAS_14_subset_1_SemiAll.csv', index=False)  # save file


        df = pd.read_csv("GW_MAIN_PFAS_14_subset_1_SemiAll.csv")

        ### oversample & train the multi-label XGB-chain with all data (include the P-label)
        model_name = 'XGBoost_chain'
        print(model_name)

        # 将数据的feature和label分开，同时将数据分成训练集合测试集
        X = df.iloc[:, 0:113]      # 114 # 111
        # X.toarray()  # convert to dense numpy array
        y = df.iloc[:, 113:]       # .values
        # get the train/test data -- # get the test data
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=2)  # random_state=2 结果可以被复现
        print('len(test)', len(xtest))

        # 1 SMOTE Oversampling for Multi-Class Classification
        print('raw dataset')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0]/y.iloc[:, i].count()
            print(i, list(y.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X, y, 0)  # Getting minority instance of that dataframe
        X, y = MLSMOTE(X, y, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 1')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0]/y.iloc[:, i].count()
            print( i, list(y.columns.values)[i], ': ', counter, 'rate:' , rate)
        ## repeat the process until the dataset is balanced
        '''
        X_sub, y_sub = get_minority_instace(X_res, y_res, 0)  # Getting minority instance of that dataframe
        X_res_1, y_res_1 = MLSMOTE(X_res, y_res, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 2')
        for i in range(y_res_1.shape[1]):
            counter = Counter(y_res_1.iloc[:, i])
            rate = counter[0]/y_res_1.iloc[:, i].count()
            print( i, list(y_res_1.columns.values)[i], ': ', counter, 'rate:' , rate)

        X_sub, y_sub = get_minority_instace(X_res_1, y_res_1, 0)  # Getting minority instance of that dataframe
        X_res_2, y_res_2 = MLSMOTE(X_res_1, y_res_1, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 3')
        for i in range(y_res_2.shape[1]):
            counter = Counter(y_res_2.iloc[:, i])
            rate = counter[0] / y_res_2.iloc[:, i].count()
            print(i, list(y_res_2.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_2, y_res_2, 1)  # Getting minority instance of that dataframe
        X_res_3, y_res_3 = MLSMOTE(X_res_2, y_res_2, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 4 -- change > ')
        for i in range(y_res_3.shape[1]):
            counter = Counter(y_res_3.iloc[:, i])
            rate = counter[0] / y_res_3.iloc[:, i].count()
            print(i, list(y_res_3.columns.values)[i], ': ', counter, 'rate:', rate)

        X_sub, y_sub = get_minority_instace(X_res_3, y_res_3, 1)  # Getting minority instance of that dataframe
        X_res_4, y_res_4 = MLSMOTE(X_res_3, y_res_3, X_sub, y_sub, 2000)  # Applying MLSMOTE to augment the dataframe
        # summarize class distribution
        print('oversampling dataset 5 -- change > ')
        for i in range(y_res_4.shape[1]):
            counter = Counter(y_res_4.iloc[:, i])
            rate = counter[0] / y_res_4.iloc[:, i].count()
            print(i, list(y_res_4.columns.values)[i], ': ', counter, 'rate:', rate)

        # oversampling - duplicated the get_minority_instace
        X_sub, y_sub = get_minority_instace(X_res_4, y_res_4, 0)  # Getting minority instance of that dataframe
        X, y = oversampling(X_res_4, y_res_4, X_sub, y_sub)  #
        # summarize class distribution
        print('oversampling dataset 6 -- < duplicated the get_minority_instace ')
        for i in range(y.shape[1]):
            counter = Counter(y.iloc[:, i])
            rate = counter[0] / y.iloc[:, i].count()
            print(i, list(y.columns.values)[i], ': ', counter, 'rate:', rate)
        '''

        # 2 train the model
        # ClassifierChain
        classifier = ClassifierChain(XGBClassifier())
        clf = Pipeline([('classify', classifier)])
        print(clf)

        clf.fit(xtrain, ytrain)
        print(clf.score(xtrain, ytrain))
        print('test score', clf.score(xtest, ytest))

        yhat = clf.predict(xtest)
        accuracy = accuracy_score(ytest, yhat)
        print('accuracy: ', accuracy)
        print('precision_score', precision_score(y_true=ytest, y_pred=yhat, average='samples'))
        print('recall_score',recall_score(y_true=ytest, y_pred=yhat, average='samples'))
        print('f1_score', f1_score(ytest, yhat, average='samples'))
        # Hamming Loss -- 衡量的是所有样本中，预测错的标签数在整个标签标签数中的占比. 其值越小表示模型的表现结果越好
        print('hamming_loss',hamming_loss(ytest, yhat))

        ytest = ytest.values
        # classifier chain
        auc_y1 = roc_auc_score(ytest[:, 0], yhat[:, 0].toarray())
        auc_y2 = roc_auc_score(ytest[:, 1], yhat[:, 1].toarray())
        auc_y3 = roc_auc_score(ytest[:, 2], yhat[:, 2].toarray())
        auc_y4 = roc_auc_score(ytest[:, 3], yhat[:, 3].toarray())
        auc_total = np.mean([auc_y1, auc_y2, auc_y3, auc_y4])
        print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4))
        print("ROC AUC total ave: ", auc_total)

        cm_y1 = confusion_matrix(ytest[:, 0], yhat[:, 0].toarray())
        cm_y2 = confusion_matrix(ytest[:, 1], yhat[:, 1].toarray())
        cm_y3 = confusion_matrix(ytest[:, 2], yhat[:, 2].toarray())
        cm_y4 = confusion_matrix(ytest[:, 3], yhat[:, 3].toarray())
        print(cm_y1, cm_y2, cm_y3, cm_y4)

        cr_y1 = classification_report(ytest[:, 0], yhat[:, 0].toarray())
        cr_y2 = classification_report(ytest[:, 1], yhat[:, 1].toarray())
        cr_y3 = classification_report(ytest[:, 2], yhat[:, 2].toarray())
        cr_y4 = classification_report(ytest[:, 3], yhat[:, 3].toarray())
        print(cr_y1, cr_y2, cr_y3, cr_y4)

        # xgboost https://dongr0510.medium.com/multi-label-classification-example-with-multioutputclassifier-and-xgboost-in-python-98c84c7d379f
        # XGBoost https://www.kaggle.com/code/para24/ovr-vs-multioutput-vs-classifier-chaining/notebook


    else:
        print('no file name found')


if __name__ == "__main__":
    main( )


print('\n--------------------------------------------------------------\n')



# Multi-Class Imbalanced Classification  https://machinelearningmastery.com/multi-class-imbalanced-classification/
#

