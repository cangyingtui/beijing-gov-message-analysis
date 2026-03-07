# -*- coding: utf-8 -*-
"""
北京市各区留言数量热力图
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 读取数据
# =========================
df = pd.read_pickle("processed_data.pkl")
map_gdf = gpd.read_file("beijing_districts_real.geojson")

print("📌 当前数据包含的字段：")
print(df.columns.tolist())

# =========================
# 2. 选择“最可靠的区级来源字段”
# =========================
if "当前的位置" in df.columns:
    source_col = "当前的位置"
elif "当前位置" in df.columns:
    source_col = "当前位置"
elif "location" in df.columns:
    source_col = "location"
else:
    raise KeyError("❌ 未找到任何可用于提取行政区的字段（当前的位置 / 当前位置 / location）")

print(f"✅ 使用字段提取行政区：{source_col}")

# =========================
# 3. 提取行政区
# =========================
DISTRICTS = [
    "东城区","西城区","朝阳区","海淀区","丰台区","石景山区",
    "门头沟区","房山区","通州区","顺义区","昌平区","大兴区",
    "怀柔区","平谷区","密云区","延庆区"
]

def extract_district(text):
    if pd.isna(text):
        return None
    for d in DISTRICTS:
        if d in str(text):
            return d
    return None

df["district"] = df[source_col].apply(extract_district)
df = df[df["district"].notna()].copy()

# =========================
# 4. 按区统计留言数量
# =========================
district_count = (
    df.groupby("district")
      .size()
      .reset_index(name="total_messages")
)

# =========================
# 5. 合并地图并绘制
# =========================
merged = map_gdf.merge(
    district_count,
    left_on="name",
    right_on="district",
    how="left"
)

missing = merged[merged["total_messages"].isna()]
if not missing.empty:
    print("⚠️ 以下区未匹配成功：")
    print(missing["name"].tolist())

fig, ax = plt.subplots(figsize=(10, 10))

merged.plot(
    column="total_messages",
    cmap="Blues",
    linewidth=0.8,
    edgecolor="#b0b0b0",
    legend=True,
    ax=ax
)

ax.set_title("北京市各区留言数量热力图", fontsize=16)
ax.axis("off")

plt.tight_layout()
plt.savefig("图像输出/13_北京市各区留言数量热力图.png", dpi=300)
plt.close()

print("✅ 北京区级热力图生成完成")
