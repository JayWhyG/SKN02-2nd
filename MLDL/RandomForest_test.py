from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
data = pd.read_csv("C:/ex/private/period_add_dataset.csv")

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

# 데이터셋 분할
train_input, test_input, train_target, test_target = train_test_split(data_set, target, test_size=0.2, random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold 교차 검증 사용
                           n_jobs=-1,  # 모든 CPU 코어 사용
                           verbose=2)

# GridSearchCV 실행
grid_search.fit(train_input, train_target.ravel())

# 최적의 하이퍼파라미터 출력
print(f'Best parameters found: {grid_search.best_params_}')

# 최적의 모델로 예측
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(test_input)

# 평가 결과 출력
print(f'Accuracy: {accuracy_score(test_target, y_pred)}')
print(classification_report(test_target, y_pred))

# 피처 중요도 추출
feature_importances = best_rf.feature_importances_
feature_names = data[['subscription_price',
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
       'age_group_45-54', 'age_group_55-65', 'period']].columns

# 피처 중요도 데이터프레임 생성
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# 중요도 순으로 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Random Forest Model')
plt.show()

# Accuracy: 0.9273587531310882
#               precision    recall  f1-score   support

#        False       0.92      0.98      0.95      2576
#         True       0.95      0.78      0.86      1017

#     accuracy                           0.93      3593
#    macro avg       0.94      0.88      0.91      3593
# weighted avg       0.93      0.93      0.93      3593