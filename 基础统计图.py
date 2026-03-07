import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
sns.set_palette(CUSTOM_COLORS)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_pickle("processed_data.pkl")
os.makedirs("图像输出", exist_ok=True)

# 01 留言类型占比
type_counts = df['message_type'].value_counts(dropna=False)
colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(type_counts))]
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    type_counts.values,
    labels=type_counts.index.astype(str),
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
min_idx = type_counts.values.argmin()
autotexts[min_idx].set_color('white')
autotexts[min_idx].set_fontweight('bold')
ax.set_title("01 留言类型占比", fontsize=16)
plt.savefig("图像输出/01_留言类型占比.png", dpi=300)
plt.close()

# 02 热门主题 Top10（最终版）
plt.figure(figsize=(12, 6))
theme_data = df['theme'].value_counts().head(10)
colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(theme_data))]
plt.bar(range(len(theme_data)), theme_data.values, color=colors)
plt.xticks(range(len(theme_data)), theme_data.index.astype(str), rotation=45)
plt.title("02 热门主题 Top10", fontsize=16)
plt.tight_layout()
plt.savefig("图像输出/02_主题Top10.png", dpi=300)
plt.close()

# 03 月度留言趋势
df_month = df[(df['message_time'] >= '2024-01-01') & (df['message_time'] < '2025-12-01')].copy()
df_month['month'] = df_month['message_time'].dt.to_period('M')
monthly = df_month.groupby('month').size()
monthly.index = monthly.index.astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(x=monthly.index, y=monthly.values, marker='o', linewidth=2.5)
plt.fill_between(monthly.index, monthly.values, alpha=0.3)
plt.title("03 月度留言趋势", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("图像输出/03_月度留言趋势.png", dpi=300)
plt.close()

# 04 留言时段热力图
pivot = df.pivot_table(index='day_of_week', columns='hour', values='message_id', aggfunc='count')
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
pivot = pivot.reindex(order)
pivot.index = ['周一','周二','周三','周四','周五','周六','周日']

cmap = LinearSegmentedColormap.from_list(
    "custom", ["#ffffff", CUSTOM_COLORS[2], CUSTOM_COLORS[1], CUSTOM_COLORS[0]]
)

plt.figure(figsize=(16, 8))
sns.heatmap(pivot, cmap=cmap)
plt.title("04 留言时段热力图", fontsize=16)
plt.tight_layout()
plt.savefig("图像输出/04_留言热力图.png", dpi=300)
plt.close()

# 05 回复单位 Top20
top_units = df["reply_unit"].value_counts().head(20)
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_units)), top_units.values, color=CUSTOM_COLORS)
plt.yticks(range(len(top_units)), top_units.index)
plt.gca().invert_yaxis()
plt.title("05 回复单位 Top20", fontsize=16)
plt.tight_layout()
plt.savefig("图像输出/05_回复单位Top20.png", dpi=300)
plt.close()

# 06 办理状态分布
status_counts = df["status"].value_counts()
colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(status_counts))]
plt.figure(figsize=(6, 6))
plt.pie(status_counts.values, labels=status_counts.index,
        autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("06 留言办理状态分布", fontsize=16)
plt.savefig("图像输出/06_留言状态分布.png", dpi=300)
plt.close()

# 07 各类留言处理时长箱线图（≤60天，最终无 warning 版）

short_df = df[df["duration_days"] <= 60].copy()

# 获取实际类别数量
message_types = short_df["message_type"].dropna().unique()
palette = CUSTOM_COLORS[:len(message_types)]

plt.figure(figsize=(14, 6))
sns.boxplot(
    data=short_df,
    x="message_type",
    y="duration_days",
    hue="message_type",
    palette=palette,
    legend=False
)

plt.title("07 各类留言处理时长（≤60天）", fontsize=16)
plt.xlabel("留言类型")
plt.ylabel("处理时长（天）")
plt.tight_layout()
plt.savefig("图像输出/07_处理时长箱线图.png", dpi=300)
plt.close()

