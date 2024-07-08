#철회수 내 구독 비율 & 구독별 철회 비율

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/ex/private/period_add_dataset.csv")

# 구독 유형별로 철회된 행의 수 계산
churn_counts_BASIC = data[data['transaction_type_CHURN'] == 1]['subscription_type_BASIC'].sum()
churn_counts_MAX = data[data['transaction_type_CHURN'] == 1]['subscription_type_MAX'].sum()
churn_counts_PRO = data[data['transaction_type_CHURN'] == 1]['subscription_type_PRO'].sum()

# 구독 유형별로 전체 행의 수 계산
total_counts_BASIC = data['subscription_type_BASIC'].sum()
total_counts_MAX = data['subscription_type_MAX'].sum()
total_counts_PRO = data['subscription_type_PRO'].sum()

# 철회 비율 계산
churn_rate_BASIC = churn_counts_BASIC / total_counts_BASIC if total_counts_BASIC != 0 else 0
churn_rate_MAX = churn_counts_MAX / total_counts_MAX if total_counts_MAX != 0 else 0
churn_rate_PRO = churn_counts_PRO / total_counts_PRO if total_counts_PRO != 0 else 0

#철회율 총합 계산
total_counts = churn_rate_BASIC + churn_rate_PRO + churn_rate_MAX

#철회율 총합 대비 비율 계산
rate_basic = churn_rate_BASIC / total_counts * 100
rate_pro = churn_rate_PRO / total_counts * 100
rate_max = churn_rate_MAX / total_counts * 100

#시각화
labels = ['BASIC', 'PRO', 'MAX']
churn_rates = [rate_basic, rate_pro, rate_max]
colors = ['#ff9999', '#66b3ff', '#99ff99']

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# 원그래프로 시각화
axs[0].pie(churn_rates, labels=labels, colors=colors, textprops={'fontsize': 15, 'color': 'black'}, autopct='%1.2f%%', startangle=0)
axs[0].set_title('Subscription Type Rate In Churn Counts')

# 바 그래프로 시각화
labels = ['BASIC', 'MAX', 'PRO']
churn_rates2 = [churn_rate_BASIC * 100, churn_rate_MAX * 100, churn_rate_PRO * 100]

axs[1].bar(labels, churn_rates2, color=colors)
axs[1].set_xlabel('Subscription Type')
axs[1].set_ylabel('Churn Rate(%)')
axs[1].set_title('Churn Rate by Subscription Type')
axs[1].set_ylim(0, 50)

for i, v in enumerate(churn_rates2):
    axs[1].text(i, v + 1, f"{v:.2f}", ha='center', fontsize=12)

fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.2)
plt.show()
