# 성별 유형별 업그레이드 & 철회 비율

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/ex/private/period_add_dataset.csv")

# 성별 유형별로 시작된 행의 수 계산
Initial_counts_Female = data[data['transaction_type_initial'] == 1]['customer_gender_Female'].sum()
Initial_counts_Male = data[data['transaction_type_initial'] == 1]['customer_gender_Male'].sum()
Initial_counts_Other = data[data['transaction_type_initial'] == 1]['customer_gender_Other'].sum()

# 성별 유형별로 업그레이드된 행의 수 계산
upgrade_counts_Female = data[data['transaction_type_UPGRADE'] == 1]['customer_gender_Female'].sum()
upgrade_counts_Male = data[data['transaction_type_UPGRADE'] == 1]['customer_gender_Male'].sum()
upgrade_counts_Other = data[data['transaction_type_UPGRADE'] == 1]['customer_gender_Other'].sum()

# 성별 유형별로 업그레이드를 하고 철회를 한 행의 수 계산
upgrade_churn_counts_Female = data[(data['transaction_type_UPGRADE'] == 1) & (data['transaction_type_CHURN'] == 1)]['customer_gender_Female'].sum()
upgrade_churn_counts_Male = data[(data['transaction_type_UPGRADE'] == 1) & (data['transaction_type_CHURN'] == 1)]['customer_gender_Male'].sum()
upgrade_churn_counts_Other = data[(data['transaction_type_UPGRADE'] == 1) & (data['transaction_type_CHURN'] == 1)]['customer_gender_Other'].sum()

# 최종 업그레이드 한 행의 수 계산
fin_upgrade_counts_Female = upgrade_counts_Female - upgrade_churn_counts_Female
fin_upgrade_counts_Male = upgrade_counts_Male - upgrade_churn_counts_Male
fin_upgrade_counts_Other = upgrade_counts_Other - upgrade_churn_counts_Other

# 업그레이드 비율 계산
upgrade_rate_Female = fin_upgrade_counts_Female / Initial_counts_Female if Initial_counts_Female != 0 else 0
upgrade_rate_Male = fin_upgrade_counts_Male / Initial_counts_Male if Initial_counts_Male != 0 else 0
upgrade_rate_Other = fin_upgrade_counts_Other / Initial_counts_Other if Initial_counts_Other != 0 else 0

# 결과 출력
print()
print("Upgrade rate for Female subscription: ", upgrade_rate_Female * 100)
print("Upgrade rate for Male subscription: ", upgrade_rate_Male * 100)
print("Upgrade rate for Other subscription: ", upgrade_rate_Other * 100)

# 성별 유형별로 철회된 행의 수 계산
churn_counts_Female = data[data['transaction_type_CHURN'] == 1]['customer_gender_Female'].sum()
churn_counts_Male = data[data['transaction_type_CHURN'] == 1]['customer_gender_Male'].sum()
churn_counts_Other = data[data['transaction_type_CHURN'] == 1]['customer_gender_Other'].sum()

print()
print(churn_counts_Female + churn_counts_Male + churn_counts_Other)

# 철회 비율 계산
churn_rate_Female = churn_counts_Female / Initial_counts_Female if Initial_counts_Female != 0 else 0
churn_rate_Male = churn_counts_Male / Initial_counts_Male if Initial_counts_Male != 0 else 0
churn_rate_Other = churn_counts_Other / Initial_counts_Other if Initial_counts_Other != 0 else 0

# 결과 출력
print()
print("Churn rate for Female subscription: ", churn_rate_Female * 100)
print("Churn rate for Male subscription: ", churn_rate_Male * 100)
print("Churn rate for Other subscription: ", churn_rate_Other * 100)

# 스택형 바 그래프로 시각화
plt.figure(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99']
labels = ['Female', 'Male', 'Other']
initial_rates = [100 - (upgrade_rate_Female * 100 + churn_rate_Female * 100),
                 100 - (upgrade_rate_Male * 100 + churn_rate_Male * 100),
                 100 - (upgrade_rate_Other * 100 + churn_rate_Other * 100)]
upgrade_rates = [upgrade_rate_Female * 100, upgrade_rate_Male * 100, upgrade_rate_Other * 100]
churn_rates = [churn_rate_Female * 100, churn_rate_Male * 100, churn_rate_Other * 100]

x = range(len(labels))

fig, ax = plt.subplots()

# 시작 및 업그레이드 스택형 바
bar1 = ax.bar(x, initial_rates, color=colors[0], label='Initial')
bar2 = ax.bar(x, upgrade_rates, color=colors[1], bottom=initial_rates, label='Upgrade')
bar3 = ax.bar(x, churn_rates, color=colors[2], bottom=[i + j for i, j in zip(initial_rates, upgrade_rates)], label='Churn')

ax.set_xlabel('Gender')
ax.set_ylabel('Rate(%)')
ax.set_title('Upgrade, and Churn Rates by Gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 값 표시
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

plt.tight_layout()
plt.show()
