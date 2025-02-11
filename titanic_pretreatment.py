# titanicの前処理を色々試すためライブラリ
# data = pd.concat([train, test], sort=False) 
# 上記で結合したデータに前処理を行った結果のdataを返す

import numpy as np
import pandas as pd

def tutrial1(data: pd.Series) -> pd.Series:
    # 性別データを0/1に置き換える
    data['Sex'] = data['Sex'].replace(['male', 'female'], [0, 1])

    # 乗船港の欠損データをSにする
    data['Embarked'] = data['Embarked'].fillna('S')

    # 乗船港(C/Q/S)を数値に置き換える
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 運賃の欠損データを平均値に置き換える
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    # 年齢の欠損データを標準偏差と平均年齢を元に補完
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    data['Age'] = data['Age'].fillna(np.random.randint(max(0, age_avg - age_std), age_avg + age_std))

    # 名前、乗客者ID, 乗船していた兄弟/配偶者の数、乗船していた両親と子供の数、チケットID、客室番号を削除する
    delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
    data = data.drop(delete_columns, axis=1)

    return data