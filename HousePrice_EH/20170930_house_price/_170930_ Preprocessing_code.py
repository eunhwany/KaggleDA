import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

train_path = "D:/python_kaggle/train.csv"
test_path = "D:/python_kaggle/test.csv"
output_path = "D:/python_kaggle/output.csv"

## train preprocessing
raw_train_data = pd.read_csv(train_path)
raw_test_data = pd.read_csv(test_path)

raw_train_data = raw_train_data.drop('Id',axis=1)
raw_test_data = raw_test_data.drop('Id',axis=1)
train_x_without_train_y = raw_train_data.drop('SalePrice', axis=1)
train_y = raw_train_data['SalePrice']

all_data = pd.concat([train_x_without_train_y, raw_test_data])

categorical = []
numerical = []

for feature in all_data.columns:
    if all_data[feature].dtype == 'object':
        categorical.append(feature)
    else:
        numerical.append(feature)

len(numerical)
len(categorical)

del(numerical[numerical.index('MSSubClass')])
categorical.append('MSSubClass')

len(numerical)
len(categorical)

nu_df = all_data[numerical]
nu_df = nu_df.fillna(0)
ca_df = pd.DataFrame()

## (수치형 변수 - 평균) / 분산  방법으로 정규화
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
nu_df = normalize(nu_df)
################################################

## Test의 카테고리가 Train에 없는 변수 찾기
mismatch = []
for element in categorical:
    train_length = pd.get_dummies(raw_train_data[element]).shape[1]
    test_length = pd.get_dummies(raw_test_data[element]).shape[1]
    if train_length != test_length:
        mismatch.append(element)
categorical = list(set(categorical)-set(mismatch))
################################################

for element in categorical:
    ca_df = pd.concat([ca_df,pd.get_dummies(all_data[element])], axis=1)

new_x = pd.concat([nu_df, ca_df], axis=1)
train_x = new_x.iloc[:1460,:]
test_x = new_x.iloc[1460:,:]

## hyper_parameter 탐색
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)
tmp = 0
score = []
hyper_param = []
for i in range(100,3000,100):
    lr = RandomForestRegressor(n_estimators=i).fit(X_train, y_train)
    model_score = lr.score(X_test,y_test)
    score.append(model_score)
    hyper_param.append(i)
    print(i)
plt.plot(hyper_param, score)

## modeling
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
lr = RandomForestRegressor(n_estimators=hyper_param[score.index(max(score))]).fit(train_x,train_y)
# lr = LinearRegression().fit(train_x, train_y)
# lr = Ridge().fit(train_x, train_y)
pred = lr.predict(test_x)

## 예측값이 모두 양수인지 확인
all(pred>0)
np.savetxt(output_path, pred)
plt.hist(lr.coef_)
# np.savetxt(output_path, nu_df, fmt='%1.1e', delimiter=',', header=' '.join(list(nu_df.columns.values)))
