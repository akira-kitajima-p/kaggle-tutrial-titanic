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

print(X_train.columns)

# 訓練を行う(ロジスティック回帰)
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# 訓練データをもとに結果を予測する
y_pred_familySize_isAlone = clf.predict(X_test)

# 結果をcsvに出力する
sub = gender_submission
sub['Survived'] = list(map(int, y_pred_familySize_isAlone))
sub.to_csv("submission_familySize_isAlone.csv", index=False)

# FamilySizeを抜いて学習し、結果をcsvに出力する
clf.fit(X_train.drop('FamilySize', axis=1), y_train)
y_pred_isAlone = clf.predict(X_test.drop('FamilySize', axis=1))
sub['Survived'] = list(map(int, y_pred_isAlone))
sub.to_csv("submission_isAlone.csv", index=False)

# IsAloneを抜いて学習し、結果をcsvに出力する
clf.fit(X_train.drop('IsAlone', axis=1), y_train)
y_pred_isAlone = clf.predict(X_test.drop('IsAlone', axis=1))
sub['Survived'] = list(map(int, y_pred_isAlone))
sub.to_csv("submission_FamilySize.csv", index=False)

# FamilySize, IsAloneを両方抜いて学習し、結果をcsvに出力する
clf.fit(X_train.drop(['FamilySize', 'IsAlone'], axis=1), y_train)
y_pred = clf.predict(X_test.drop(['FamilySize', 'IsAlone'], axis=1))
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index = False)
