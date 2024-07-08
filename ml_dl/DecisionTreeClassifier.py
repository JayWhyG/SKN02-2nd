import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/ex/private/period_add_dataset.csv")
Churn = data[data['transaction_type_CHURN']==True]['cust_id'].unique()
data['churn_result'] = data['cust_id'].apply(lambda x: True if x in Churn else False)

data_set = data[['subscription_price',
       'subscription_type_BASIC', 'subscription_type_MAX',
       'subscription_type_PRO', 'customer_country_Denmark',
       'customer_country_Finland', 'customer_country_Norway',
       'customer_country_Sweden', 'transaction_type_CHURN',
       'transaction_type_REDUCTION', 'transaction_type_UPGRADE',
       'transaction_type_initial', 'customer_gender_Female',
       'customer_gender_Male', 'customer_gender_Other', 'referral_type_Bing',
       'referral_type_Display', 'referral_type_Google Ads',
       'referral_type_Organic Search', 'referral_type_Paid Search',
       'referral_type_TV', 'referral_type_Unknown', 'referral_type_facebook',
       'age_group_18-24', 'age_group_25-34', 'age_group_35-44',
       'age_group_45-54', 'age_group_55-65', 'period']].to_numpy()
target = data[['churn_result']].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data_set, target, test_size= 0.2, random_state= 42)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report

## 하이퍼파라미터 설정
param_grid = {
    'max_depth': [3, 4, 5, 7, 10, 13, 16],
    'min_samples_split': [2, 3, 4, 5, 6, 7],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']}

# 모델 초기화
dt = DecisionTreeClassifier()

# 그리드 서치 실행
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(train_input, train_target)

# 최적 하이퍼파라미터 출력
print(f'Best Parameters: {grid_search.best_params_}')

# 최적 하이퍼파라미터로 모델 학습
best_params = grid_search.best_params_
best_dt = DecisionTreeClassifier(**best_params)
best_dt.fit(train_input, train_target)

# 테스트 셋 평가
y_pred = best_dt.predict(test_input)

# 평가 결과 출력
print(f'Accuracy: {accuracy_score(test_target, y_pred)}')
print(classification_report(test_target, y_pred))

# Decision Tree 출력
plt.figure(dpi=5000)
plot_tree(best_dt, filled = True)
plt.show()

# Decision Tree
# Accuracy: 0.8883940996381854
#               precision    recall  f1-score   support

#        False       0.89      0.97      0.93      2576
#         True       0.89      0.69      0.78      1017

#    accuracy                            0.89      3593
#    macro avg       0.89      0.83      0.85      3593
# weighted avg       0.89      0.89      0.88      3593