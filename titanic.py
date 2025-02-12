import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from titanic_pretreatment import tutrial3

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
gender_submission = pd.read_csv('./gender_submission.csv')

# 学習データと訓練データを結合する(特徴量エンジニアリングの為)
data = pd.concat([train, test], sort=False)
data = tutrial3(data)

# 訓練データとテストデータを前処理を行ったデータに置き換える
train = data[:len(train)]
test = data[len(train):]

# 学習データを入力値(x_train)と出力値(y_train)にわける
y_train = train['Survived']
X_train = train.drop('Survived', axis = 1)
# テストデータからカラムを削除する（当然全てNaNなので要らない）
X_test = test.drop('Survived', axis = 1)

def drop_train_predict(X_train, y_train, X_test, drop_columns=[]):
    """
    指定されたカラムを削除したデータでロジスティック回帰を学習し、予測を行う
    Args:
        X_train (pd.DataFrame): 学習用の特徴量データ
        y_train (pd.Series): 学習用のラベルデータ
        X_test (pd.DataFrame): テストデータ
        drop_columns (list): 学習前に削除するカラムのリスト
    Returns:
        np.array: 予測結果
    """
    X_train_mod = X_train.drop(columns=drop_columns, errors='ignore')
    X_test_mod = X_test.drop(columns=drop_columns, errors='ignore')

    clf = LogisticRegression(penalty='l2', solver="sag", random_state=0, max_iter=10000)
    clf.fit(X_train_mod, y_train)
    return clf.predict(X_test_mod)

# 予測パターンと出力ファイル名を定義
drop_patterns = {
    "submission_familySize_isAlone.csv": [],
    "submission_isAlone.csv": ["FamilySize"],
    "submission_FamilySize.csv": ["IsAlone"],
    "submission.csv": ["FamilySize", "IsAlone"]
}

# ループで学習と予測を実行
sub = gender_submission
for filename, drops in drop_patterns.items():
    sub['Survived'] = drop_train_predict(X_train, y_train, X_test, drops)
    sub.to_csv(filename, index=False)
