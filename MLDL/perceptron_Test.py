import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_set2)
target2 = data[['churn_result']]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target2, test_size=0.2, random_state=42)

# 다층 퍼셉트론 모델 구축
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 요약
model.summary()

# 배치 크기를 다르게 설정하여 모델 학습
batch_sizes = [32, 64, 128, 256]

for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=2)
    print(batch_size)
    
    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Batch Size: {batch_size} - Accuracy: {accuracy}')
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # 평가 결과 출력
    print(classification_report(y_test, y_pred_classes))


#     Batch Size: 256 - Accuracy: 0.9446145296096802
# 113/113 ━━━━━━━━━━━━━━━━━━━━ 0s 473us/step
#               precision    recall  f1-score   support

#        False       0.94      0.99      0.96      2576
#         True       0.96      0.84      0.90      1017

#     accuracy                           0.94      3593
#    macro avg       0.95      0.91      0.93      3593
# weighted avg       0.95      0.94      0.94      3593

