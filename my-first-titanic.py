# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
plt.style.use('ggplot')

train_csv_path = "../input/train.csv"
train_df = pd.read_csv(train_csv_path)
train_df.describe()
train_df.isnull().sum()

def preprocess_sex(df):
    sex_dum = pd.get_dummies(df['Sex'])
    df = pd.concat((df,sex_dum),axis=1)
    df = df.drop(['Sex', 'female'],axis=1)
    df.rename(columns={'male':'Sex'},inplace=True)
    return df

def preprocess_embarked(df):
    emb_dum = pd.get_dummies(df['Embarked'])
    df = pd.concat((df,emb_dum),axis=1)
    df = df.drop(['Embarked', 'S'],axis=1)
    return df

def fill_age_each_name(df):
    mr_re = ".*, Mr\..*"
    mrs_re = ".*, Mrs\..*"
    miss_re = ".*, Miss\..*"
    master_re = ".*, Master\..*"
    name_re_list = [mr_re, mrs_re, miss_re, master_re]
    median_list = [find_median(df, name_re) for name_re in name_re_list]
    for name_re, median in zip(name_re_list, median_list):
        fill_age(df, name_re, median)
    return df

def find_median(df, name_re):
    name_df = df[df["Name"].str.contains(name_re)]
    median = name_df.dropna().Age.median()
    return median

def fill_age(df, name_re, median):
    df.loc[(df.Age.isnull()) & (df["Name"].str.contains(name_re)), "Age"] = median

def fill_age_non_category(df):
    age_median = df.groupby('Sex').Age.median()
    df.loc[(df.Age.isnull()) & (df["Sex"] == 0), "Age"] = age_median[0]
    df.loc[(df.Age.isnull()) & (df["Sex"] == 1), "Age"] = age_median[1]

def draw(df):
    df_sur = df[df.Survived==1]
    df_sur_age = df_sur.loc[:,"Age"]
    df_sur_s = df_sur.loc[:,"Sex"]
    plt.scatter(df_sur_age,df_sur_s,color="#cc6699",alpha=0.5)

    df_dead = df[df.Survived==0]
    df_dead_age = df_dead.loc[:,"Age"]
    df_dead_s = df_dead.loc[:,"Sex"]
    plt.scatter(df_dead_age,df_dead_s,color="#6699cc",alpha=0.5)

    plt.show()

def logistic_regression(X, Y):
    model = LogisticRegression()
    model.fit(X, Y)
    print("a:", model.coef_, "b:", model.intercept_)
    print("R^2:", model.score(X, Y))
    return model

def cross_validation(model, X, Y):
    cross_val_score_list = []
    for i in range(10):
        kfold = KFold(n_splits=5, shuffle=True)
        cross_val_score_list.append(cross_val_score(model, X, Y, cv=kfold, n_jobs=1))
    cross_val_list = [cross_val_score.mean() for cross_val_score in cross_val_score_list]
    cross_val_list
    print(np.mean(cross_val_list))

def export_file():
    test_df = pd.read_csv(test_csv_path)
    result_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':np.array(test_predict)})
    result_csv_path = "/Users/yoshinori/development/training/py_training/titanic/result.csv"
    result_df.to_csv(result_csv_path, index=False)

train_df = preprocess_sex(train_df)
train_df = preprocess_embarked(train_df)
# draw(train_df)

train_df = fill_age_each_name(train_df)
fill_age_non_category(train_df)

X = train_df.loc[:,["Sex","Age","Pclass","SibSp","Parch"]]
Y = train_df.loc[:,"Survived"]

model = logistic_regression(X, Y)
cross_validation(model, X, Y)

test_csv_path = "../input/test.csv"
test_df = pd.read_csv(test_csv_path)
test_df = preprocess_sex(test_df)
test_df = preprocess_embarked(test_df)
test_df = fill_age_each_name(test_df)
fill_age_non_category(test_df)

test_df = test_df.loc[:,["Sex","Age","Pclass","SibSp","Parch"]]
test_predict = model.predict(test_df)
print(test_predict)

