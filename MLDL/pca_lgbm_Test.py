import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt

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

# 데이터 분할 (PCA 데이터 사용)
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data_set, target, test_size=0.2, random_state=42)

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
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# 모델 초기화
lgb_model = lgb.LGBMClassifier()

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=LGBMClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold 교차 검증 사용
                           n_jobs=-1,  # 모든 CPU 코어 사용
                           verbose=2)

# GridSearchCV 실행
grid_search.fit(data_pca, train_target.ravel())

# 최적 하이퍼파라미터로 모델 학습
best_params = grid_search.best_params_
best_lgb = lgb.LGBMClassifier(**best_params)
best_lgb.fit(data_pca, train_target)

# 테스트 셋 평가
y_pred = best_lgb.predict(test_pca)


# 최적의 하이퍼파라미터 출력
print(f'Best parameters found: {grid_search.best_params_}')

# 평가 결과 출력
print(f'Accuracy: {accuracy_score(test_target, y_pred)}')
print(classification_report(test_target, y_pred))

# 중요 피처 시각화
lgb.plot_importance(best_lgb, max_num_features=10)
plt.show()

