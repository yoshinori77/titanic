# %matplotlib inline
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
plt.style.use('ggplot')

def preprocess_sex(df):
    sex_dum = pd.get_dummies(df['Sex'], drop_first=True)
    df = pd.concat((df,sex_dum),axis=1)
    df = df.drop(['Sex'],axis=1)
    df.rename(columns={'male':'Sex'},inplace=True)
    return df

def preprocess_embarked(df):
    emb_dum = pd.get_dummies(df['Embarked'], drop_first=True)
    df = pd.concat((df,emb_dum),axis=1)
    df = df.drop(['Embarked'],axis=1)
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
    return df

def find_title(df, name_re):
    title_series = \
        df["Name"].str.contains(name_re).apply(lambda x: 1 if x is True else 0)
    return title_series

def preprocess_title(df):
    mr_re = ".*, Mr\..*"
    mrs_re = ".*, Mrs\..*"
    miss_re = ".*, Miss\..*"
    master_re = ".*, Master\..*"
    name_re_list = [mr_re, mrs_re, miss_re, master_re]
    title_list = [find_title(df, name_re) for name_re in name_re_list]
    title_df = pd.concat(title_list, axis=1)
    title_df.columns = ['Mr', 'Mrs', 'Miss', 'Master']
    df = pd.concat([df, title_df], axis=1)
    return df

def preprocess_alone(df):
    alone = ((df['SibSp'] == 0) & (df['Parch'] == 0)).apply(lambda x: 1 if x is True else 0)
    df = pd.concat([df, alone], axis=1)
    columns = df.columns.tolist()
    columns[-1] = 'Alone'
    df.columns = columns
    return df


def preprocess(df):
    df = preprocess_sex(df)
    df = preprocess_embarked(df)
    df = fill_age_each_name(df)
    df = fill_age_non_category(df)
    df = preprocess_title(df)
    df = preprocess_alone(df)
    return df

def draw(df):
    sur_df = df[df.Survived==1]
    sur_age_df = sur_df.loc[:,"Age"]
    sur_s_df = sur_df.loc[:,"Sex"]
    plt.scatter(sur_age_df,sur_s_df,color="#cc6699",alpha=0.5)

    dead_df = df[df.Survived==0]
    dead_age_df = dead_df.loc[:,"Age"]
    dead_df_s = dead_df.loc[:,"Sex"]
    plt.scatter(dead_age_df,dead_df_s,color="#6699cc",alpha=0.5)

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
        kfold = KFold(n_splits=3, shuffle=True)
        cross_val_score_list.append(cross_val_score(model, X, Y, cv=kfold, n_jobs=1))
    cross_val_list = [cross_val_score.mean() for cross_val_score in cross_val_score_list]
    print(np.mean(cross_val_list))

def print_importance(model, X):
    feature_list = list(zip(X.columns, model.feature_importances_))
    feature_list = sorted(feature_list, key=lambda x: x[1], reverse=True)
    print(feature_list)
    feature_names = [f[0] for f in feature_list]
    return feature_names

def export_file(test_predict):
    test_df = pd.read_csv(test_csv_path)
    result_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':np.array(test_predict)})
    result_csv_path = "/Users/yoshinori/development/training/py_training/titanic/result.csv"
    result_df.to_csv(result_csv_path, index=False)

train_csv_path = "../input/train.csv"
train_df = pd.read_csv(train_csv_path)
# train_df.describe()
# sex_ct = pd.crosstab(train_df['Sex'], train_df['Survived'])
# sex_ct
# train_df.isnull().sum()
train_df.corr()
# train_df.info()
train_df.head()
# 学習
train_df = preprocess(train_df)
train_df.corr()

X_df = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
Y_df = train_df['Survived']

# draw(train_df)
# train_df.info()
# train_df[['Fare', 'Survived']].sort_values(by='Fare', ascending=False)
# train_df.info()

# train_X = train_df.loc[:,["Sex","Age","Pclass","SibSp"]]
# train_Y = train_df.loc[:,"Survived"]

# model = logistic_regression(train_X, train_Y)
# cross_validation(model, train_X, train_Y)
# model = SVC(C=3, gamma=0.1)
# model.fit(train_X, train_Y)

clf = RandomForestClassifier(max_depth=20, random_state=0).fit(X_df, Y_df)
feature_names = print_importance(clf, X_df)
X_df = X_df.drop(feature_names[-3:], axis=1)

train_X, val_X, train_Y, val_Y = train_test_split(X_df, Y_df, test_size=0.2, random_state=0)


clf = SVC(kernel='linear', C=1).fit(train_X, train_Y)
print(u"再代入誤り率：", 1 - clf.score(train_X, train_Y))
print(u"ホールドアウト誤り率：", 1 - clf.score(val_X, val_Y))

# clf = SVC(kernel='rbf', C=1)
# clf = SVC(kernel='linear', C=5)
# clf = RandomForestClassifier(max_depth=25, random_state=0)
clf = GradientBoostingClassifier(max_depth=3, random_state=0, learning_rate=0.003,  n_estimators=500)
scores = cross_val_score(clf, train_X, train_Y, cv=3,)
print("scores: ", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GradientBoostingClassifier(max_depth=3, random_state=0, learning_rate=0.003,  n_estimators=500).fit(train_X, train_Y)
print(u"再代入誤り率：", 1 - clf.score(train_X, train_Y))
print(u"ホールドアウト誤り率：", 1 - clf.score(val_X, val_Y))

# テスト
test_csv_path = "../input/test.csv"
test_df = pd.read_csv(test_csv_path)
test_df = preprocess(test_df)
test_X = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin']+feature_names[-3:], axis=1)

# test_X = test_df.loc[:,["Sex","Age","Pclass","SibSp"]]
test_X = test_X.fillna(0)
test_predict = clf.predict(test_X)
print(test_predict)

# ファイル書き出し
# export_file(test_predict)
