import pickle

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pip._vendor.webencodings import labels
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR



def evaluate_predictions(predictions, true):

    mae = np.mean(abs(predictions - true))

    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    return mae, rmse


def evaluate(X_train, X_test, y_train, y_test):
    # model list
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                       'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')

    # define model
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=50)

    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)

    # training model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]

    # baseline
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    return results




if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font='SimHei')
    student = pd.read_csv('student-mat-pass-or-fail.csv') # get information from dataset
    G3 = student[['G3']]
    failures = student[['failures']]
    list = G3.values.tolist()
    list2 = failures.values.tolist()
    G3list=[]
    failures_list=[]
    for i in range(0, 395):
        G3list.append(list[i][0])
    for i in range(0, 395):
        failures_list.append(list2[i][0])
    print(G3.describe())
    labels = student['G3']

    # student = student.drop(['school','G1','G2'],axis='columns')

    student = pd.get_dummies(student)

    most_correlated = student.corr().abs()['G3'].sort_values(ascending=False)

    most_correlated = most_correlated[:6]

    print(most_correlated)

    # cutting dateset
    X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size=0.75, random_state=42)
    # results = evaluate(X_train, X_test, y_train, y_test)
    # print(results)
    # X_train = X_train[['failures', 'Medu', 'higher', 'pass', 'age']]
    # X_test = X_test[['failures', 'Medu', 'higher', 'pass', 'age']]
    X_train = X_train[['failures','Medu','G1','pass','G2']] # use high correlation attributes
    X_test = X_test[['failures','Medu','G1','pass','G2']]

    print(X_train)
    print(X_test)
    test = student[['failures','Medu','G1','pass','G2']]
    print(test)

    regr = LinearRegression()   # use training done LinearRegression model
    regr.fit(X_train, y_train)
    a = regr.predict(X_test)
    print(a)
    b = regr.score(X_test, y_test)
    print("degree of fitting")
    print(b)
    c = regr.predict(test)
    print(c)
    filename = 'LR_Model with 5 features include G1 and G2'
    pickle.dump(regr, open(filename, 'wb'))



