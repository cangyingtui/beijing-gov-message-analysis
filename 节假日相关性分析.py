import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 自定义配色（与之前文件保持一致）
CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
sns.set_palette(CUSTOM_COLORS)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data():
    """加载并准备数据"""
    df = pd.read_pickle("processed_data.pkl")

    # 只保留有回复的数据
    df = df.dropna(subset=['duration_days'])
    df = df[df['duration_days'] > 0]

    return df


def define_holidays():
    """定义2024-2025年的中国法定节假日"""
    holidays = {
        # 2024年节假日
        '2024-01-01': '元旦',
        '2024-02-10': '春节', '2024-02-11': '春节', '2024-02-12': '春节',
        '2024-02-13': '春节', '2024-02-14': '春节', '2024-02-15': '春节',
        '2024-02-16': '春节', '2024-02-17': '春节',
        '2024-04-04': '清明', '2024-04-05': '清明', '2024-04-06': '清明',
        '2024-05-01': '劳动节', '2024-05-02': '劳动节', '2024-05-03': '劳动节',
        '2024-05-04': '劳动节', '2024-05-05': '劳动节',
        '2024-06-08': '端午', '2024-06-09': '端午', '2024-06-10': '端午',
        '2024-09-15': '中秋', '2024-09-16': '中秋', '2024-09-17': '中秋',
        '2024-10-01': '国庆', '2024-10-02': '国庆', '2024-10-03': '国庆',
        '2024-10-04': '国庆', '2024-10-05': '国庆', '2024-10-06': '国庆',
        '2024-10-07': '国庆',

        # 2025年节假日（预测日期，需根据实际调整）
        '2025-01-01': '元旦',
        '2025-01-28': '春节', '2025-01-29': '春节', '2025-01-30': '春节',
        '2025-01-31': '春节', '2025-02-01': '春节', '2025-02-02': '春节',
        '2025-02-03': '春节',
        '2025-04-04': '清明', '2025-04-05': '清明', '2025-04-06': '清明',
        '2025-05-01': '劳动节', '2025-05-02': '劳动节', '2025-05-03': '劳动节',
        '2025-05-04': '劳动节', '2025-05-05': '劳动节',
        '2025-05-31': '端午', '2025-06-01': '端午', '2025-06-02': '端午',
        '2025-10-01': '国庆', '2025-10-02': '国庆', '2025-10-03': '国庆',
        '2025-10-04': '国庆', '2025-10-05': '国庆', '2025-10-06': '国庆',
        '2025-10-07': '国庆'
    }

    return pd.DataFrame(list(holidays.items()), columns=['date', 'holiday_name'])


def create_holiday_features(df, holidays_df):
    """创建节假日相关特征"""
    # 将日期转换为datetime格式
    df['message_date'] = df['message_time'].dt.date
    holidays_df['date'] = pd.to_datetime(holidays_df['date']).dt.date

    # 创建节假日映射
    holiday_dates = set(holidays_df['date'])
    holiday_name_map = dict(zip(holidays_df['date'], holidays_df['holiday_name']))

    # 计算距离最近节假日的天数
    def days_to_nearest_holiday(message_date):
        message_dt = pd.Timestamp(message_date)
        min_days = float('inf')

        for holiday_date in holiday_dates:
            holiday_dt = pd.Timestamp(holiday_date)
            days_diff = (holiday_dt - message_dt).days

            # 取绝对值最小的天数
            if abs(days_diff) < abs(min_days):
                min_days = days_diff

        return min_days

    def get_holiday_period(days_diff):
        """根据距离节假日的天数分类"""
        if days_diff > 7:
            return '节假日后>7天'
        elif days_diff > 0:
            return '节假日后1-7天'
        elif days_diff == 0:
            return '节假日当天'
        elif days_diff >= -3:
            return '节假日前1-3天'
        else:
            return '节假日前>3天'

    # 计算特征
    df['days_to_holiday'] = df['message_date'].apply(days_to_nearest_holiday)
    df['holiday_period'] = df['days_to_holiday'].apply(get_holiday_period)

    # 标记是否节假日期间
    df['is_holiday'] = df['message_date'].isin(holiday_dates)

    # 标记是否周末
    df['is_weekend'] = df['message_time'].dt.dayofweek >= 5

    # 组合特征：节假日类型
    df['day_type'] = '工作日'
    df.loc[df['is_holiday'], 'day_type'] = '法定假日'
    df.loc[df['is_weekend'] & (~df['is_holiday']), 'day_type'] = '周末'

    return df


def analyze_holiday_response_time(df):
    """分析节假日与回复时长的关系"""
    # 1. 不同节假日期间的回复时长统计
    holiday_stats = df.groupby('holiday_period').agg({
        'duration_days': ['mean', 'median', 'std', 'count']
    }).round(2)

    holiday_stats.columns = ['平均天数', '中位数', '标准差', '样本数']
    holiday_stats = holiday_stats.sort_values('平均天数', ascending=False)

    # 2. 按日期类型统计
    day_type_stats = df.groupby('day_type').agg({
        'duration_days': ['mean', 'median', 'std', 'count']
    }).round(2)

    day_type_stats.columns = ['平均天数', '中位数', '标准差', '样本数']
    day_type_stats = day_type_stats.sort_values('平均天数', ascending=False)

    return holiday_stats, day_type_stats


def create_heatmap_data(df):
    """创建热力图所需数据"""
    # 按节假日期间和月份分析回复时长
    df['message_month'] = df['message_time'].dt.strftime('%Y-%m')

    # 筛选数据范围
    months = sorted(df['message_month'].unique())

    # 创建透视表
    heatmap_data = df.pivot_table(
        index='holiday_period',
        columns='message_month',
        values='duration_days',
        aggfunc='mean',
        fill_value=0
    )

    # 重新排序行
    period_order = ['节假日前>3天', '节假日前1-3天', '节假日当天',
                    '节假日后1-7天', '节假日后>7天']
    heatmap_data = heatmap_data.reindex(period_order)

    # 选择最近的12个月展示
    recent_months = months[-12:] if len(months) > 12 else months
    heatmap_data = heatmap_data[recent_months]

    return heatmap_data


def plot_holiday_analysis(df, holiday_stats, day_type_stats, heatmap_data):
    """绘制节假日分析图表"""
    # 创建输出目录
    import os
    os.makedirs("图像输出", exist_ok=True)

    fig = plt.figure(figsize=(18, 12))

    # 1. 节假日期间平均回复时长柱状图
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(holiday_stats.index, holiday_stats['平均天数'],
                   color=CUSTOM_COLORS[:len(holiday_stats)])

    # 添加数值标签
    for bar, val in zip(bars, holiday_stats['平均天数']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax1.set_title('01 不同节假日期间的平均回复时长', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均回复时长（天）')
    ax1.set_xlabel('节假日期间分类')
    plt.xticks(rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # 2. 日期类型平均回复时长柱状图
    ax2 = plt.subplot(2, 2, 2)
    bars2 = ax2.bar(day_type_stats.index, day_type_stats['平均天数'],
                    color=CUSTOM_COLORS[:len(day_type_stats)])

    for bar, val in zip(bars2, day_type_stats['平均天数']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    ax2.set_title('02 不同日期类型的平均回复时长', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均回复时长（天）')
    ax2.set_xlabel('日期类型')
    plt.xticks(rotation=0)
    ax2.grid(True, alpha=0.3)

    # 3. 节假日期间回复时长分布箱线图
    ax3 = plt.subplot(2, 2, 3)

    # 准备数据
    box_data = []
    labels = []
    for period in ['节假日前>3天', '节假日前1-3天', '节假日当天',
                   '节假日后1-7天', '节假日后>7天']:
        if period in df['holiday_period'].unique():
            period_data = df[df['holiday_period'] == period]['duration_days']
            # 限制异常值
            period_data = period_data[period_data <= period_data.quantile(0.95)]
            if len(period_data) > 0:
                box_data.append(period_data)
                labels.append(period)

    box = ax3.boxplot(box_data, labels=labels, patch_artist=True)

    # 设置箱线图颜色
    for patch, color in zip(box['boxes'], CUSTOM_COLORS[:len(box_data)]):
        patch.set_facecolor(color)

    ax3.set_title('03 节假日期间回复时长分布（箱线图）', fontsize=14, fontweight='bold')
    ax3.set_ylabel('回复时长（天）')
    ax3.set_xlabel('节假日期间分类')
    plt.xticks(rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # 4. 节假日期间与月份的热力图
    ax4 = plt.subplot(2, 2, 4)

    if not heatmap_data.empty:
        # 创建自定义渐变色
        cmap = LinearSegmentedColormap.from_list(
            "custom_holiday",
            [CUSTOM_COLORS[4], CUSTOM_COLORS[2], CUSTOM_COLORS[0]]
        )

        # 绘制热力图
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            ax=ax4,
            cbar_kws={'label': '平均回复时长（天）'}
        )

        ax4.set_title('04 节假日期间与月份的热力图分析', fontsize=14, fontweight='bold')
        ax4.set_xlabel('月份')
        ax4.set_ylabel('节假日期间分类')

    plt.suptitle('节假日与回复时长关联度分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('图像输出/节假日回复时长分析.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 额外：详细热力图（单独保存）
    if not heatmap_data.empty:
        fig, ax = plt.subplots(figsize=(15, 8))

        # 使用seaborn的热力图
        sns.heatmap(
            heatmap_data,
            cmap=LinearSegmentedColormap.from_list(
                "holiday_heatmap",
                [CUSTOM_COLORS[4], CUSTOM_COLORS[2], CUSTOM_COLORS[0]]
            ),
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': '平均回复时长（天）', 'shrink': 0.8}
        )

        ax.set_title('节假日期间与月份的热力图分析（详细版）', fontsize=16, fontweight='bold')
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('节假日期间分类', fontsize=12)

        plt.tight_layout()
        plt.savefig('图像输出/节假日热力图_详细版.png', dpi=300, bbox_inches='tight')
        plt.show()


def create_summary_table(df, holiday_stats, day_type_stats):
    """创建汇总统计表"""
    # 整体统计
    overall_stats = pd.DataFrame({
        '指标': ['总留言数', '有回复留言数', '平均回复时长(天)', '中位数回复时长(天)',
                 '最长回复时长(天)', '最短回复时长(天)'],
        '数值': [
            len(df),
            len(df.dropna(subset=['duration_days'])),
            round(df['duration_days'].mean(), 2),
            round(df['duration_days'].median(), 2),
            round(df['duration_days'].max(), 2),
            round(df['duration_days'].min(), 2)
        ]
    })

    # 保存统计表
    with pd.ExcelWriter('图像输出/节假日分析统计表.xlsx', engine='openpyxl') as writer:
        overall_stats.to_excel(writer, sheet_name='整体统计', index=False)
        holiday_stats.to_excel(writer, sheet_name='节假日期间统计')
        day_type_stats.to_excel(writer, sheet_name='日期类型统计')

    print("✅ 统计表已保存至: 图像输出/节假日分析统计表.xlsx")


def main():
    """主函数"""
    print("开始分析节假日与回复时长的关联度...")

    # 1. 加载数据
    df = load_and_prepare_data()
    print(f"📊 加载数据完成，共 {len(df)} 条有回复记录")

    # 2. 定义节假日
    holidays_df = define_holidays()
    print(f"📅 定义 {len(holidays_df)} 个节假日")

    # 3. 创建节假日特征
    df = create_holiday_features(df, holidays_df)

    # 4. 分析数据
    holiday_stats, day_type_stats = analyze_holiday_response_time(df)

    # 5. 创建热力图数据
    heatmap_data = create_heatmap_data(df)

    # 6. 打印关键发现
    print("\n📈 关键发现:")
    print(f"1. 最长平均回复时长: {holiday_stats['平均天数'].idxmax()} ({holiday_stats['平均天数'].max():.1f}天)")
    print(f"2. 最短平均回复时长: {holiday_stats['平均天数'].idxmin()} ({holiday_stats['平均天数'].min():.1f}天)")
    print(
        f"3. 节假日当天 vs 节假日前>3天 差异: {(holiday_stats.loc['节假日当天', '平均天数'] - holiday_stats.loc['节假日前>3天', '平均天数']):.1f}天")

    # 7. 绘制图表
    plot_holiday_analysis(df, holiday_stats, day_type_stats, heatmap_data)

    # 8. 保存统计表
    create_summary_table(df, holiday_stats, day_type_stats)

    print("\n✅ 分析完成！")
    print("📁 生成文件:")
    print("   - 图像输出/节假日回复时长分析.png")
    print("   - 图像输出/节假日热力图_详细版.png")
    print("   - 图像输出/节假日分析统计表.xlsx")


if __name__ == "__main__":
    main()