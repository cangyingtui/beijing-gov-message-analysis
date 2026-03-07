import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4']

df = pd.read_pickle("processed_data.pkl")

# 提取区级字段
def extract_district(location):
    if pd.isna(location):
        return "未知区域"
    match = re.search(r"(?:北京市)?(.*?区)", location)
    return match.group(1) if match else "未知区域"

df["district"] = df["location"].apply(extract_district)
df["duration_days"] = df["duration_days"].apply(lambda x: x if x is not None and x >= 0 else None)
df["is_overtime"] = df["duration_days"] > 15

# ✅ 2. 聚合生成统计数据
summary = df.groupby("district").agg(
    total_messages=("message_id", "count"),
    avg_duration=("duration_days", "mean"),
    percent_overtime=("is_overtime", lambda x: round(100 * x.sum() / x.count(), 1))
).reset_index()

def get_top_reply_info(x):
    most_common = x.value_counts()
    return pd.Series({
        "reply_unit_top1": most_common.idxmax(),
        "reply_unit_top1_ratio": round(most_common.max() / most_common.sum(), 2)
    })

top_units = df.groupby("district")["reply_unit"].apply(get_top_reply_info).reset_index()
final_df = summary.merge(top_units, on="district", how="left")
final_df.to_csv("district_summary.csv", index=False, encoding="utf-8-sig")
print("✅ 统计数据已保存为 district_summary.csv")

# ✅ 3. 读取真实北京地图（GeoJSON）
map_gdf = gpd.read_file("北京市_县.geojson")  # 文件将由我提供
merged = map_gdf.merge(final_df, left_on="name", right_on="district")

# ✅ 4. 绘图
fig, ax = plt.subplots(figsize=(10, 10))
cmap = LinearSegmentedColormap.from_list("custom_blues", CUSTOM_COLORS)
merged.plot(
    column="total_messages",
    cmap=cmap,
    linewidth=0.8,
    ax=ax,
    edgecolor="0.7",
    legend=True
)
plt.title("北京各区留言数量热力图", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.savefig("北京留言热力图_真实地图版.png", dpi=300)
plt.show()
print("✅ 地图已保存为 北京留言热力图_真实地图版.png")
