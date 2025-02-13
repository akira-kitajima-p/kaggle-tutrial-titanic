import numpy as np
import pandas as pd
from titanic_pretreatment import tutrial3
from predict import drop_train_predict

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
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

# 予測パターンと出力ファイル名を定義
drop_patterns = {
    # "submission_familySize_isAlone.csv": [],
    # "submission_isAlone.csv": ["FamilySize"],
    # "submission_FamilySize.csv": ["IsAlone"],
    "submission.csv": ["FamilySize", "IsAlone"]
}

# 使いたいモデルを変更可能（"lr", "rf", "lgbm"）
model_type = "rf"

# ループで学習と予測を実行
sub = gender_submission
for filename, drops in drop_patterns.items():
    sub['Survived'] = drop_train_predict(X_train, y_train, X_test, model_type, drops)
    sub.to_csv(filename.replace("submission", f"{model_type}_submission"), index=False)
