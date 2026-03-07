import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os

CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
sns.set_palette(CUSTOM_COLORS)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
font_prop = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")

os.makedirs("图像输出", exist_ok=True)

df = pd.read_pickle("processed_data.pkl")
df = df[(df['message_time'] >= '2024-01-01') & (df['message_time'] < '2025-12-01')]
df['month'] = df['message_time'].dt.to_period('M')

# 11 月度留言量趋势
monthly_counts = df.groupby("month").size()
monthly_counts.index = monthly_counts.index.astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o', linewidth=2.5)
plt.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.3)
plt.title("11 月度留言数量趋势", fontproperties=font_prop, fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("图像输出/11_月度留言趋势.png", dpi=300)
plt.close()

# 12 主题 × 月份堆叠面积图
pivot_df = df.pivot_table(index='month', columns='theme',
                          values='message_id', aggfunc='count').fillna(0)
pivot_df.index = pivot_df.index.astype(str)

plt.figure(figsize=(16, 8))
ax = pivot_df.plot(
    kind='area',
    stacked=True,
    colormap="Blues",
    alpha=0.9,
    figsize=(16, 8)
)

ax.legend(
    title="主题",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

plt.title("12 各主题留言趋势堆叠图", fontproperties=font_prop, fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("图像输出/12_主题堆叠趋势图.png", dpi=300)
plt.close()

# 13 平均处理时长趋势
duration_monthly = df.groupby("month")["duration_days"].mean()
duration_monthly.index = duration_monthly.index.astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(x=duration_monthly.index, y=duration_monthly.values,
             marker='o', linewidth=2.5, color=CUSTOM_COLORS[3])
plt.fill_between(duration_monthly.index, duration_monthly.values,
                 alpha=0.3, color=CUSTOM_COLORS[3])
plt.title("13 月度平均处理时长趋势", fontproperties=font_prop, fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("图像输出/13_处理时长趋势图.png", dpi=300)
plt.close()
