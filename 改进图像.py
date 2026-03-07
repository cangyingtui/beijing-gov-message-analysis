# ==========================================
# 改进图像.py ｜最终稳定版 V5（版式终稿）
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

df = pd.read_pickle("processed_data_with_district.pkl")


COLOR_PALETTE = {
    "blue_dark":  (32/255, 56/255, 136/255),
    "blue":       (81/255, 141/255, 219/255),
    "blue_light": (167/255, 210/255, 228/255),
    "yellow":     (245/255, 215/255, 163/255),
    "orange":     (225/255, 156/255, 102/255)
}


df["is_handled"] = df["reply_time"].notna()

OVERTIME_DAYS = 30
df["is_overtime"] = False
df.loc[
    (df["is_handled"]) & (df["duration_days"] > OVERTIME_DAYS),
    "is_overtime"
] = True


def plot_theme_message_type_stack(df):
    cnt = (
        df.groupby(["theme", "message_type"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    cnt["ratio"] = cnt["count"] / cnt.groupby("theme")["count"].transform("sum")
    pivot = cnt.pivot(index="theme", columns="message_type", values="ratio")

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        color=[
            COLOR_PALETTE["blue_dark"],
            COLOR_PALETTE["blue"],
            COLOR_PALETTE["yellow"],
            COLOR_PALETTE["orange"]
        ]
    )

    ax.set_title("不同主题下留言类型结构对比", fontsize=14)
    ax.set_xlabel("主题")
    ax.set_ylabel("比例")

    plt.legend(title="留言类型", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_reply_unit_capacity(df, top_n=20, label_n=8):
    """
    top_n  : 参与绘图的回复单位数量（按工作量排序）
    label_n: 图中显示名称的重点单位数量
    """

    agg = (
        df.groupby("reply_unit")
          .agg(
              count=("reply_unit", "size"),
              avg_days=("duration_days", "mean"),
              overtime_rate=("is_overtime", "mean")
          )
          .sort_values("count", ascending=False)
          .head(top_n)
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    # 所有点（背景）
    ax.scatter(
        agg["count"],
        agg["avg_days"],
        s=agg["overtime_rate"] * 800 + 80,
        color=COLOR_PALETTE["orange"],
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5
    )

    # 重点单位（只标前 label_n 个）
    highlight = agg.head(label_n)

    for _, row in highlight.iterrows():
        ax.text(
            row["count"],
            row["avg_days"],
            row["reply_unit"],
            fontsize=9,
            ha="left",
            va="bottom"
        )

    ax.set_xlabel("工作量（留言数量）")
    ax.set_ylabel("平均处理时长（天）")
    ax.set_title("回复单位综合能力评估（重点单位标注）")

    # 手动图例说明
    legend_text = (
        "气泡大小：超期率\n"
        f"仅标注工作量前 {label_n} 的回复单位"
    )

    fig.text(
        0.98, 0.02,
        legend_text,
        ha="right",
        va="bottom",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_theme_message_type_stack(df)
    plot_reply_unit_capacity(df)
