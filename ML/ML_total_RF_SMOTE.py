# binary classification  --  RandomForestClassifier
# SMOTE - 合成少数类过采样技术
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, r2_score  # accuracy
from sklearn.metrics import mean_squared_error    # Mean Squared Error (MSE)
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics                 # f-i score
from sklearn.metrics import accuracy_score
import random
import math

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import imblearn       # 使用不平衡学习 Python 库提供的实现
print(imblearn.__version__)

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN   # 自适应合成采样 (ADASYN)

# 调参
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 二分类数据的ROC曲线可视化
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
from sklearn.metrics import *


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # add the parameters needed to change
    parser.add_argument("--which_data", type=str, default='pfas_rf',help='pfas_rf,pfas_data,pfas_data,example')
    return parser

def main( ):

    # parameters
    parser = config_parser()
    args = parser.parse_args()

    file_name = args.which_data
    print('file_name',file_name)

    if file_name == 'example':
        print('example dataset -- make')
        ### 用于平衡数据的 SMOTE
        # 生成和绘制合成二元分类问题
        # define dataset
        X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
        # summarize class distribution
        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()

        # 接下来使用 SMOTE 对少数类进行过采样并绘制转换后的数据集
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        # summarize the new class distribution
        counter = Counter(y)
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()

    elif file_name == 'pfas_data':
        print('pfas dataset')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_pfas_total.csv")
        # Training And Test Data -- drop location cols
        df = df.drop(columns=['gm_gis_dwr_region'])     # 'latitude','longitude',
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        #y = df.iloc[:, 113:].values        # .values for XBBoost, catb, lightb
        y = np.array(df["Label_PFAS_total"])


        # 1 SMOTE Oversampling for Binary classification
        # summarize class distribution
        counter = Counter(y)   # Counter({0: 19121, 1: 7779})
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()

        # SMOTE 对少数类进行过采样并绘制转换后的数据集
        print('SMOTE')
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)        # 0.9741142633023925

        # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
        print('ADASYN')
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        y = np.array(df["Label_PFAS_total"])
        oversample = ADASYN()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)

        # 首先使用 SMOTE 对少数类进行过采样到大约 1:10 (sampling_strategy=0.1) 的比例，然后对多数类进行欠采样以达到大约 1:2 (sampling_strategy=0.5)的比例
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        y = np.array(df["Label_PFAS_total"])             # Counter({0: 19120, 1: 7779})
        under = RandomUnderSampler(sampling_strategy=0.5)
        X, y = under.fit_resample(X, y)
        counter = Counter(y)    # Counter({0: 15558, 1: 7779})
        print(counter)
        oversample = SMOTE(sampling_strategy=0.95)
        X, y = oversample.fit_resample(X, y)
        counter = Counter(y)    # Counter({0: 15558, 1: 14780})
        print(counter)
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train The Random Forest
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        # Apply RandomForestRegressor To Test Data
        y_pred = clf.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)
        # train_score = clf.score(X_train, y_train)
        # test_score = clf.score(X_test, y_test)
        # r2 = r2_score(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmae = np.sqrt(mean_squared_error(y_test, y_pred))
        # print('TEST Score： ', test_score)
        # print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    elif file_name == 'pfas_rf':
        print('pfas ml training')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_pfas_total.csv")
        # Training And Test Data -- drop location cols
        df = df.drop(columns=['latitude','longitude','gm_gis_dwr_region'])     # 'latitude','longitude',

        # change date to year & month -- Timestamp To Date
        # df['Year'] = pd.DatetimeIndex(df['date']).year
        # df['Year'] = df['date'].dt.year
        # df['Month'] = df['date'].dt.month
        # df = df.drop(columns=['date'])  # drop col

        X = df.iloc[:, 0:112].values      # delete .values for rf importance  [new:112 -- no lat/long/date + year/month]
        #y = df.iloc[:, 113:].values        # .values for XBBoost, catb, lightb
        y = np.array(df["Label_PFAS_total"])

        # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
        print('ADASYN')
        oversample = ADASYN()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train The Random Forest
        # RF -- ML optimization
        print('Random forest')

        # search for the best n_estimators
        '''
        # n_estimators是影响程度最大的参数，我们先对其进行调整 # 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
        score_lt = []
        # 每隔10步建立一个随机森林，获得不同n_estimators的得分
        for i in range(0, 200, 10):
            rfc = RandomForestClassifier(n_estimators=i + 1, random_state=90)
            score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
            score_lt.append(score)
            print(i)
        score_max = max(score_lt)
        print('score_max：{}'.format(score_max),
              'n_estimators：{}'.format(score_lt.index(score_max) * 10 + 1))   # n_estimators 子树数量

        # 绘制学习曲线  --  find the best n_estimators
        x = np.arange(1, 241, 10)
        plt.subplot(111)
        plt.plot(x, score_lt, 'r-')
        plt.show()
        '''
        # 接下来的调参方向是使模型复杂度减小的方向，从而接近泛化误差最低点。我们使用能使模型复杂度减小，并且影响程度排第二的max_depth。
        '''
        # 建立n_estimators为200的随机森林
        rfc = RandomForestClassifier(n_estimators=200, random_state=90)

        # 用网格搜索调整max_depth
        param_grid = {'max_depth': np.arange(1, 20)}
        GS = GridSearchCV(rfc, param_grid, cv=10)
        GS.fit(X_train, y_train)

        best_param = GS.best_params_
        best_score = GS.best_score_
        print(best_param, best_score)

        # 用网格搜索调整max_features
        param_grid = {'max_features': np.arange(20, 51)}

        rfc = RandomForestClassifier(n_estimators=200
                                     , random_state=90
                                     , max_depth=52)
        GS = GridSearchCV(rfc, param_grid, cv=10)
        GS.fit(X_train, y_train)
        best_param = GS.best_params_
        best_score = GS.best_score_
        print(best_param, best_score)
        '''

        # RF - after optimization
        clf = RandomForestClassifier(n_estimators=200,max_depth=52,max_features=35,random_state=1)  #  n_estimators=200  max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        # Apply RandomForestRegressor To Test Data
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)
        # train_score = clf.score(X_train, y_train)
        # test_score = clf.score(X_test, y_test)
        # r2 = r2_score(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmae = np.sqrt(mean_squared_error(y_test, y_pred))
        # print('TEST Score： ', test_score)
        # print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # 结果可视化
        # AUROC Curve  -- https://zhuanlan.zhihu.com/p/364400255

        ## 可视化在验证集上的Roc曲线
        pre_y = clf.predict_proba(X_test)[:, 1]
        fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
        aucval = auc(fpr_Nb, tpr_Nb)  # 计算auc的取值
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_Nb, tpr_Nb, "r", linewidth=3)
        plt.grid()
        plt.xlabel("False Postive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("RF ROC curve")
        plt.text(0.15, 0.9, "AUC = " + str(round(aucval, 4)))
        plt.show()


        '''
        # 将训练集结果可视化
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        # 将测试集结果可视化
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        '''


        # importance
        col = list(X_train.columns.values)
        importances = clf.feature_importances_
        col_list = df.columns.values.tolist()
        x_columns = col_list[0:-1]
        # print("importances：", importances)
        # Returns the index value of the array from largest to smallest
        indices = np.argsort(importances)[::-1]
        list01 = []
        list02 = []
        for f in range(X_train.shape[1]):
            # For the final need to be sorted in reverse order, I think it is to do a value similar to decision tree backtracking,
            # from the leaf to the root, the root is more important than the leaf.
            print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
            list01.append(col[indices[f]])
            list02.append(importances[indices[f]])

        c = {"columns": list01, "importances": list02}
        data_impts = DataFrame(c)
        # data_impts.to_excel('RF_data_importances.xlsx')

        importances = list(clf.feature_importances_)
        feature_list = list(X_train.columns)

        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        x_values = list(range(len(importances)))
        print(x_values)


    # reviewer's comments:
    elif file_name == 'pfas_ABN':
        print('pfas ml training')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_pfas_total.csv")
        # Training And Test Data -- drop location cols
        df = df.drop(columns=['latitude','longitude','gm_gis_dwr_region'])     # 'latitude','longitude',

        # change date to year & month -- Timestamp To Date
        # df['Year'] = pd.DatetimeIndex(df['date']).year
        # df['Year'] = df['date'].dt.year
        # df['Month'] = df['date'].dt.month
        # df = df.drop(columns=['date'])  # drop col

        X = df.iloc[:, 0:112].values      # delete .values for rf importance  [new:112 -- no lat/long/date + year/month]
        #y = df.iloc[:, 113:].values        # .values for XBBoost, catb, lightb
        y = np.array(df["Label_PFAS_total"])

        # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
        print('ADASYN')
        oversample = ADASYN()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


if __name__ == "__main__":
    main( )


print('\n--------------------------------------------------------------\n')



# https://zhuanlan.zhihu.com/p/440648816
# RF 调参 -- https://zhuanlan.zhihu.com/p/126288078


