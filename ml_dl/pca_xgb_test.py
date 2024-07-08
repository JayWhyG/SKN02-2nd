import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 데이터 로드
data = pd.read_csv("C:/ex/private/period_add_dataset.csv")

# PCA에 사용할 데이터 준비
data_set2 = data[['subscription_price',
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
       'age_group_45-54', 'age_group_55-65', 'period']]
target = data[['churn_result']]

# 데이터 분할 (PCA 데이터 사용)
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data_set2, target, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
data_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.fit_transform(test_input)

# PCA 적용
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
test_pca = pca.fit_transform(test_scaled)


# 하이퍼파라미터 그리드 설정
param_grid = {
    'max_depth': [9, 10, 11, 12, 13, 14, 15],
    'learning_rate': [0.15, 0.17, 0.19, 0.2, 0.22, 0.24, 0.26],
    'n_estimators': [200, 300, 350, 400, 500],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.9, 1.0]
}

# XGBoost 모델 초기화 및 학습 (PCA 데이터 사용)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 그리드 서치 실행 (5-fold 교차 검증)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(data_pca, train_target)

# 최적 하이퍼파라미터로 모델 학습
best_params = grid_search.best_params_
best_xgb = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
best_xgb.fit(data_pca, train_target)


# 최적 하이퍼파라미터 출력
print(f'Best Parameters: {grid_search.best_params_}')

# 예측 및 평가 (PCA 데이터 사용)
y_pred = best_xgb.predict(test_pca)
print(f'Accuracy (PCA Data): {accuracy_score(test_target, y_pred)}')
print(classification_report(test_target, y_pred))


# Accuracy (Original Data): 0.9499025883662677
#               precision    recall  f1-score   support

#        False       0.94      0.99      0.97      2576
#         True       0.98      0.84      0.91      1017

#     accuracy                           0.95      3593
#    macro avg       0.96      0.92      0.94      3593
# weighted avg       0.95      0.95      0.95      3593