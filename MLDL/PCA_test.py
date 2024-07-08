import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 데이터 로드
data = pd.read_csv("C:/ex/private/period_add_dataset.csv")

# PCA에 사용할 데이터 준비
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

# 데이터 스케일링
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_set)

# PCA 수행
pca = PCA(n_components=2)  # 2개의 주성분으로 축소
principal_components = pca.fit_transform(data_scaled)

# PCA 결과를 DataFrame으로 변환
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['churn_result'] = data['churn_result']
print(pca_df)

# 분산 설명률 출력
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by PC1 and PC2: {explained_variance}')

# 2D Scatter Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='churn_result', data=pca_df, palette='coolwarm')
plt.title('PCA of Customer Data')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% Variance)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% Variance)')
plt.legend(title='Churn Result')
plt.grid(True)
plt.show()
