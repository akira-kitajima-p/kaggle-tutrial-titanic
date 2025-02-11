import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
gender_submission = pd.read_csv('./gender_submission.csv')

# 学習データと訓練データを結合する(特徴量エンジニアリングの為)
data = pd.concat([train, test], sort=False)

# 性別データを0/1に置き換える
data['Sex'].replace(['male','female'], [0, 1], inplace=True)

# 乗船港(C/Q/S)の欠損データをSにする
data['Embarked'].fillna(('S'), inplace=True)
# 乗船港(C/Q/S)を数値に置き換える
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# 運賃の欠損データを平均値に置き換える
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

# 年齢の欠損データを標準偏差と平均年齢を元に良い感じの分布で置き換える
age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

# 名前、乗客者ID, 乗船していた兄弟/配偶者の数、乗船していた両親と子供の数、チケットID、客室番号を削除する
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

# 訓練データとテストデータを前処理を行ったデータに置き換える
train = data[:len(train)]
test = data[len(train):]

# 学習データを入力値(x_train)と出力値(y_train)にわける
y_train = train['Survived']
X_train = train.drop('Survived', axis = 1)
# テストデータからカラムを削除する（当然全てNaNなので要らない）
X_test = test.drop('Survived', axis = 1)

# 訓練を行う(ロジスティック回帰)
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)

# 訓練データをもとに結果を予測する
y_pred = clf.predict(X_test)

# 結果をcsvに出力する
sub = gender_submission
sub['Survived'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)

