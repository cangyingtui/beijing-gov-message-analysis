import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import os

# ===== 风格设置 =====
CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
sns.set_palette(CUSTOM_COLORS)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
font_prop = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")

os.makedirs("图像输出", exist_ok=True)

# ===== 数据 =====
df = pd.read_pickle("processed_data.pkl")
df = df[(df['message_time'] >= '2024-01-01') & (df['message_time'] < '2025-12-01')].copy()

# ===== 08 各主题平均处理时长 =====
theme_counts = df['theme'].value_counts()
valid_themes = theme_counts[theme_counts > 50].index

data = (
    df[df['theme'].isin(valid_themes)]
    .groupby('theme')['duration_days']
    .mean()
    .sort_values()
)

plt.figure(figsize=(14, 8))

norm = plt.Normalize(data.min(), data.max())
cmap = LinearSegmentedColormap.from_list(
    "efficiency",
    [CUSTOM_COLORS[0], CUSTOM_COLORS[1], "#dbe9f6", "#f3e3cf"]
)
bar_colors = [cmap(norm(v)) for v in data.values]

ax = sns.barplot(
    x=data.values,
    y=data.index,
    hue=data.index,
    palette=bar_colors,
    legend=False
)

plt.title("08 各主题平均处理时长（样本量 > 50）", fontproperties=font_prop, fontsize=16)
plt.xlabel("平均处理时长（天）", fontproperties=font_prop)
plt.ylabel("主题", fontproperties=font_prop)

for i, v in enumerate(data.values):
    ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontproperties=font_prop)

plt.tight_layout()
plt.savefig("图像输出/08_主题处理效率排行.png", dpi=300)
plt.close()

# ===== 09 排名表 =====
duration_by_theme = df.groupby('theme')['duration_days'].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table_data = []
themes = duration_by_theme.index.tolist()
durations = duration_by_theme.values.tolist()

for i in range(0, len(themes), 2):
    row = [
        themes[i], f"{durations[i]:.1f}",
        themes[i+1] if i+1 < len(themes) else "",
        f"{durations[i+1]:.1f}" if i+1 < len(durations) else ""
    ]
    table_data.append(row)

table = ax.table(
    cellText=table_data,
    colLabels=["主题", "平均处理时长", "主题", "平均处理时长"],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

plt.title("09 各主题处理时长排名表", fontproperties=font_prop, fontsize=16)
plt.tight_layout()
plt.savefig("图像输出/09_处理时长排行表.png", dpi=300)
plt.close()
