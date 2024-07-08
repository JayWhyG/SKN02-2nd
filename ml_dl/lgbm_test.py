import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

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

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data_set, target, test_size= 0.2, random_state= 42)

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

# 그리드 서치 실행 (5-fold 교차 검증)
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(train_input, train_target)

# 최적 하이퍼파라미터 출력
print(f'Best Parameters: {grid_search.best_params_}')

# 최적 하이퍼파라미터로 모델 학습
best_params = grid_search.best_params_
best_lgb = lgb.LGBMClassifier(**best_params)
best_lgb.fit(train_input, train_target)

# 테스트 셋 평가
y_pred = best_lgb.predict(test_input)
y_pred_proba = best_lgb.predict_proba(test_input)[:, 1]

# 평가 결과 출력
print(f'Accuracy: {accuracy_score(test_target, y_pred)}')
print(classification_report(test_target, y_pred))



# ROC 곡선과 AUC 계산
test_target_1d = test_target.ravel()  # 1차원 배열로 변환
fpr, tpr, _ = roc_curve(test_target_1d, y_pred_proba)
roc_auc = auc(fpr, tpr)

# ROC 곡선 시각화
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.text(0.75, 0.1, f'AUC = {roc_auc:.2f}', fontsize=20, bbox=dict(facecolor='white', alpha=0.6))
plt.grid(True)
plt.show()

# 평가 결과 출력
print(f'Accuracy: {accuracy_score(test_target, best_lgb.predict(test_input))}')
print(classification_report(test_target, best_lgb.predict(test_input)))

# Accuracy: 0.9512941831338714
#               precision    recall  f1-score   support

#        False       0.94      1.00      0.97      2576
#         True       0.99      0.84      0.91      1017

#     accuracy                           0.95      3593
#    macro avg       0.96      0.92      0.94      3593
# weighted avg       0.95      0.95      0.95      3593
