import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings('ignore')

CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
sns.set_palette(CUSTOM_COLORS)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据"""
    df = pd.read_pickle("processed_data.pkl")
    return df


def define_holiday_periods():
    # 2024-2025年主要节假日
    holidays = {
        # 2024年
        '2024元旦': ('2023-12-30', '2024-01-01'),
        '2024春节': ('2024-02-09', '2024-02-17'),
        '2024清明': ('2024-04-04', '2024-04-06'),
        '2024劳动节': ('2024-05-01', '2024-05-05'),
        '2024端午': ('2024-06-08', '2024-06-10'),
        '2024中秋': ('2024-09-15', '2024-09-17'),
        '2024国庆': ('2024-10-01', '2024-10-07'),

        # 2025年
        '2025元旦': ('2024-12-30', '2025-01-01'),
        '2025春节': ('2025-01-28', '2025-02-03'),
        '2025清明': ('2025-04-04', '2025-04-06'),
        '2025劳动节': ('2025-05-01', '2025-05-05'),
        '2025端午': ('2025-05-31', '2025-06-02'),
        '2025国庆': ('2025-10-01', '2025-10-07'),

        # 2026 年
        '2026元旦': ('2025-12-30', '2026-01-01'),
        '2026春节': ('2026-02-16', '2026-02-22'),
        '2026清明': ('2026-04-04', '2026-04-06'),
        '2026劳动节': ('2026-05-01', '2026-05-05'),
        '2026端午': ('2026-06-19', '2026-06-21'),
        '2026中秋': ('2026-09-24', '2026-09-26'),
        '2026国庆': ('2026-10-01', '2026-10-07'),

        # 2027 年
        '2027元旦': ('2026-12-30', '2027-01-01'),
        '2027春节': ('2027-02-05', '2027-02-11'),
        '2027清明': ('2027-04-03', '2027-04-05'),
        '2027劳动节': ('2027-05-01', '2027-05-05'),
        '2027端午': ('2027-06-09', '2027-06-11'),
        '2027中秋': ('2027-09-14', '2027-09-16'),
        '2027国庆': ('2027-10-01', '2027-10-07')
    }

    return holidays


def add_holiday_features(df, holidays):
    """添加节假日特征"""
    # 确保日期格式
    df['message_date'] = pd.to_datetime(df['message_time']).dt.date
    df['date'] = pd.to_datetime(df['message_date'])

    # 标记节假日期间
    df['holiday_name'] = None
    df['is_holiday_period'] = False

    for holiday_name, (start_date, end_date) in holidays.items():
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df['date'] >= start) & (df['date'] <= end)
        df.loc[mask, 'holiday_name'] = holiday_name
        df.loc[mask, 'is_holiday_period'] = True

    # 标记节假日前7天和后7天
    df['holiday_context'] = '非节假日'

    for holiday_name, (start_date, end_date) in holidays.items():
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 节假日前7天
        pre_start = start - pd.Timedelta(days=7)
        pre_end = start - pd.Timedelta(days=1)
        pre_mask = (df['date'] >= pre_start) & (df['date'] <= pre_end)
        df.loc[pre_mask, 'holiday_context'] = f'{holiday_name[5:]}前7天'

        # 节假日后7天
        post_start = end + pd.Timedelta(days=1)
        post_end = end + pd.Timedelta(days=7)
        post_mask = (df['date'] >= post_start) & (df['date'] <= post_end)
        df.loc[post_mask, 'holiday_context'] = f'{holiday_name[5:]}后7天'

    return df


def analyze_holiday_message_volume(df):
    """分析节假日期间留言数量"""
    # 按节假日分组统计
    holiday_counts = df[df['holiday_name'].notna()].groupby('holiday_name').agg({
        'message_id': 'count',
        'duration_days': 'mean'
    }).round(2)

    holiday_counts = holiday_counts.rename(columns={
        'message_id': '留言数量',
        'duration_days': '平均回复时长'
    }).sort_values('留言数量', ascending=False)

    # 节假日与非节假日对比
    holiday_vs_normal = df.groupby('is_holiday_period').agg({
        'message_id': 'count',
        'duration_days': 'mean'
    }).round(2)

    holiday_vs_normal = holiday_vs_normal.rename(columns={
        'message_id': '留言数量',
        'duration_days': '平均回复时长'
    })

    # 节假日前中后对比
    context_counts = df.groupby('holiday_context').agg({
        'message_id': 'count',
        'duration_days': 'mean'
    }).round(2)

    context_counts = context_counts.rename(columns={
        'message_id': '留言数量',
        'duration_days': '平均回复时长'
    })

    return holiday_counts, holiday_vs_normal, context_counts


def analyze_holiday_themes(df, top_n=10):
    """分析节假日期间主题分布"""
    # 获取所有主题的总体分布
    all_themes = df['theme'].value_counts()

    # 获取节假日期间的主题分布
    holiday_themes = df[df['is_holiday_period']]['theme'].value_counts()

    # 获取非节假日期间的主题分布
    normal_themes = df[~df['is_holiday_period']]['theme'].value_counts()

    # 获取热门主题
    top_themes = all_themes.head(top_n).index.tolist()

    # 创建对比数据框
    theme_comparison = pd.DataFrame({
        '总占比': all_themes / all_themes.sum(),
        '节假日占比': holiday_themes / holiday_themes.sum(),
        '非节假日占比': normal_themes / normal_themes.sum()
    }).fillna(0)

    # 只保留热门主题
    theme_comparison = theme_comparison[theme_comparison.index.isin(top_themes)]

    # 计算节假日特殊指数（节假日占比 / 总占比）
    theme_comparison['节假日特殊指数'] = (theme_comparison['节假日占比'] / theme_comparison['总占比']).round(3)

    # 筛选出节假日期间显著增加的主题
    increased_themes = theme_comparison[theme_comparison['节假日特殊指数'] > 1.2].sort_values('节假日特殊指数',
                                                                                              ascending=False)

    return theme_comparison, increased_themes


def create_theme_comparison_chart(df, theme_comparison, holiday_vs_normal):
    """创建主题对比图表 - 优化文字排版版本"""
    import os
    os.makedirs("图像输出", exist_ok=True)

    # 计算绝对数量而非百分比
    total_holiday = holiday_vs_normal.loc[True, '留言数量']
    total_normal = holiday_vs_normal.loc[False, '留言数量']

    # 计算各主题在节假日和非节假日的实际数量
    holiday_theme_counts = df[df['is_holiday_period']]['theme'].value_counts()
    normal_theme_counts = df[~df['is_holiday_period']]['theme'].value_counts()

    # 只保留热门主题
    top_themes = theme_comparison.index.tolist()
    holiday_top_counts = holiday_theme_counts[holiday_theme_counts.index.isin(top_themes)]
    normal_top_counts = normal_theme_counts[normal_theme_counts.index.isin(top_themes)]

    # 创建数据框
    theme_counts_df = pd.DataFrame({
        '节假日': holiday_top_counts,
        '非节假日': normal_top_counts
    }).fillna(0).astype(int)

    # 按节假日数量排序
    theme_counts_df = theme_counts_df.sort_values('节假日', ascending=False)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 10))

    # 设置柱状图位置
    x = np.arange(len(theme_counts_df))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width / 2, theme_counts_df['节假日'], width,
                   label='节假日', color=CUSTOM_COLORS[0], alpha=0.8)
    bars2 = ax.bar(x + width / 2, theme_counts_df['非节假日'], width,
                   label='非节假日', color=CUSTOM_COLORS[2], alpha=0.8)

    # 添加数值标签 - 优化位置
    def add_labels(bars, position_offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                va_position = 'bottom' if height < max(theme_counts_df.max()) * 0.1 else 'top'
                y_offset = 0.5 if va_position == 'top' else -0.5

                ax.text(bar.get_x() + bar.get_width() / 2 + position_offset,
                        height + y_offset,
                        f'{int(height)}',
                        ha='center', va=va_position,
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    add_labels(bars1)
    add_labels(bars2)

    # 计算并标注比例
    for i, theme in enumerate(theme_counts_df.index):
        holiday_count = theme_counts_df.loc[theme, '节假日']
        normal_count = theme_counts_df.loc[theme, '非节假日']

        if holiday_count > 0 and normal_count > 0:
            ratio = holiday_count / normal_count
            # 在柱状图上方添加比例标注
            ax.text(i, max(holiday_count, normal_count) + max(theme_counts_df.max()) * 0.05,
                    f'比例: {ratio:.2f}',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # 设置图表属性
    ax.set_xlabel('主题', fontsize=12, fontweight='bold')
    ax.set_ylabel('留言数量', fontsize=12, fontweight='bold')
    ax.set_title('节假日 vs 非节假日主题留言数量对比', fontsize=16, fontweight='bold', pad=20)

    # 优化x轴标签 - 旋转角度并调整对齐方式
    ax.set_xticks(x)
    ax.set_xticklabels(theme_counts_df.index, rotation=45, ha='right', fontsize=11)

    # 添加图例
    ax.legend(fontsize=12, loc='upper right')

    # 添加网格线
    ax.grid(True, alpha=0.3, axis='y')

    # 添加表格形式的数据总结
    # 创建表格数据
    table_data = []
    for theme in theme_counts_df.index:
        holiday_count = theme_counts_df.loc[theme, '节假日']
        normal_count = theme_counts_df.loc[theme, '非节假日']
        holiday_pct = (holiday_count / total_holiday * 100) if total_holiday > 0 else 0
        normal_pct = (normal_count / total_normal * 100) if total_normal > 0 else 0
        ratio = holiday_count / normal_count if normal_count > 0 else float('inf')

        table_data.append([
            theme,
            f'{holiday_count} ({holiday_pct:.1f}%)',
            f'{normal_count} ({normal_pct:.1f}%)',
            f'{ratio:.2f}'
        ])

    # 在图表下方添加表格
    col_labels = ['主题', '节假日留言数(占比)', '非节假日留言数(占比)', '比例(节假日/非节假日)']

    # 调整表格位置和大小
    table = plt.table(cellText=table_data,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='bottom',
                      bbox=[0, -0.7, 1, 0.5])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # 设置表头样式
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 调整布局，为表格留出空间
    plt.subplots_adjust(bottom=0.4, top=0.9)

    # 保存图表
    plt.savefig('图像输出/节假日主题数量对比_优化版.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 额外保存一个不带表格的简化版本
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # 绘制简化的柱状图
    x2 = np.arange(len(theme_counts_df))
    width2 = 0.35

    bars3 = ax2.bar(x2 - width2 / 2, theme_counts_df['节假日'], width2,
                    label='节假日', color=CUSTOM_COLORS[0], alpha=0.8)
    bars4 = ax2.bar(x2 + width2 / 2, theme_counts_df['非节假日'], width2,
                    label='非节假日', color=CUSTOM_COLORS[2], alpha=0.8)

    # 添加简洁的数值标签
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, height,
                         f'{int(height)}', ha='center', va='bottom',
                         fontsize=9, fontweight='bold')

    # 设置图表属性
    ax2.set_xlabel('主题', fontsize=12, fontweight='bold')
    ax2.set_ylabel('留言数量', fontsize=12, fontweight='bold')
    ax2.set_title('节假日 vs 非节假日主题留言数量对比（简化版）',
                  fontsize=16, fontweight='bold')

    # 优化x轴标签
    ax2.set_xticks(x2)
    ax2.set_xticklabels(theme_counts_df.index, rotation=45, ha='right', fontsize=11)

    # 添加图例和网格
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('图像输出/节假日主题数量对比_简化版.png', dpi=300, bbox_inches='tight')
    plt.show()

    return theme_counts_df


def plot_holiday_message_analysis(df, holiday_counts, holiday_vs_normal, context_counts, theme_comparison,
                                  increased_themes):
    """绘制节假日留言分析图表 - 主图"""
    import os
    os.makedirs("图像输出", exist_ok=True)

    # 创建大图
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])

    # 1. 节假日留言数量排行榜（柱状图）
    ax1 = plt.subplot(gs[0, :])
    holiday_counts_sorted = holiday_counts.sort_values('留言数量', ascending=True)
    bars = ax1.barh(holiday_counts_sorted.index, holiday_counts_sorted['留言数量'],
                    color=CUSTOM_COLORS[:len(holiday_counts)])

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + max(holiday_counts_sorted['留言数量']) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', ha='left', va='center', fontsize=10)

    ax1.set_title('01 各节假日期间留言数量排行榜', fontsize=14, fontweight='bold')
    ax1.set_xlabel('留言数量')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. 节假日与非节假日对比（分组柱状图）
    ax2 = plt.subplot(gs[1, 0])
    categories = ['节假日', '非节假日']
    message_counts = [holiday_vs_normal.loc[True, '留言数量'],
                      holiday_vs_normal.loc[False, '留言数量']]

    bars2 = ax2.bar(categories, message_counts,
                    color=[CUSTOM_COLORS[0], CUSTOM_COLORS[2]])

    # 添加数值标签 - 调整位置避免重叠
    for bar, val in zip(bars2, message_counts):
        height = bar.get_height()
        # 将标签放在柱状图内部顶部，避免外部重叠
        ax2.text(bar.get_x() + bar.get_width() / 2., height * 0.95,
                 f'{int(val)}', ha='center', va='top', fontsize=11,
                 color='white', fontweight='bold')

    ax2.set_title('02 节假日 vs 非节假日留言量', fontsize=14, fontweight='bold')
    ax2.set_ylabel('留言数量')
    ax2.grid(True, alpha=0.3)

    # 3. 平均回复时长对比（分组柱状图）
    ax3 = plt.subplot(gs[1, 1])
    response_times = [holiday_vs_normal.loc[True, '平均回复时长'],
                      holiday_vs_normal.loc[False, '平均回复时长']]

    bars3 = ax3.bar(categories, response_times,
                    color=[CUSTOM_COLORS[1], CUSTOM_COLORS[3]])

    # 添加数值标签 - 调整位置
    for bar, val in zip(bars3, response_times):
        height = bar.get_height()
        # 根据数值大小调整标签位置
        y_pos = height + 0.1 if height < max(response_times) * 0.8 else height * 0.95
        va_pos = 'bottom' if height < max(response_times) * 0.8 else 'top'
        color = 'black' if height < max(response_times) * 0.8 else 'white'

        ax3.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{val:.1f}天', ha='center', va=va_pos, fontsize=11,
                 color=color, fontweight='bold')

    ax3.set_title('03 节假日 vs 非节假日回复时长', fontsize=14, fontweight='bold')
    ax3.set_ylabel('平均回复时长（天）')
    ax3.grid(True, alpha=0.3)

    # 4. 节假日前中后留言数量变化（折线图）
    ax4 = plt.subplot(gs[1, 2])

    context_data = context_counts[~context_counts.index.str.contains('非节假日')]

    # 按节假日分组
    contexts = context_data.index.tolist()
    message_counts_context = context_data['留言数量'].values

    ax4.plot(contexts, message_counts_context, marker='o',
             linewidth=2.5, color=CUSTOM_COLORS[0])
    ax4.fill_between(contexts, message_counts_context, alpha=0.2, color=CUSTOM_COLORS[0])

    # 标记最高点和最低点
    max_idx = np.argmax(message_counts_context)
    min_idx = np.argmin(message_counts_context)
    ax4.scatter(contexts[max_idx], message_counts_context[max_idx],
                color='red', s=100, zorder=5)
    ax4.scatter(contexts[min_idx], message_counts_context[min_idx],
                color='green', s=100, zorder=5)

    ax4.set_title('04 节假日前中后留言数量变化', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间阶段')
    ax4.set_ylabel('留言数量')
    ax4.grid(True, alpha=0.3)

    # 优化x轴标签 - 旋转角度并调整间距
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    # 5. 节假日特殊主题分析（条形图）
    ax5 = plt.subplot(gs[2, :])

    if not increased_themes.empty:
        # 只取前8个最显著的主题
        top_increased = increased_themes.head(8)

        # 创建数据
        themes = top_increased.index.tolist()
        indices = top_increased['节假日特殊指数'].values

        bars5 = ax5.barh(themes, indices, color=CUSTOM_COLORS[1])

        # 添加数值标签
        for bar, idx in zip(bars5, indices):
            width = bar.get_width()
            # 将标签放在条形图内部右侧，避免外部重叠
            ax5.text(width * 0.95, bar.get_y() + bar.get_height() / 2,
                     f'{idx:.2f}', ha='right', va='center', fontsize=10,
                     color='white', fontweight='bold')

        # 添加参考线
        ax5.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='基准线(1.0)')
        ax5.axvline(x=1.2, color='orange', linestyle='--', alpha=0.5, label='显著线(1.2)')

        ax5.set_title('05 节假日期间显著增加的主题（节假日特殊指数 = 节假日占比 / 总占比）',
                      fontsize=14, fontweight='bold')
        ax5.set_xlabel('节假日特殊指数')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('图像输出/节假日留言波动分析.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_holiday_summary_report(df, holiday_counts, holiday_vs_normal, context_counts,
                                  theme_comparison, increased_themes):
    """创建节假日分析总结报告"""

    # 关键指标
    total_messages = len(df)
    holiday_messages = len(df[df['is_holiday_period']])
    holiday_percentage = (holiday_messages / total_messages) * 100

    # 找出留言最多的节假日
    busiest_holiday = holiday_counts.iloc[0]
    busiest_holiday_name = holiday_counts.index[0]

    # 找出节假日特殊指数最高的主题
    if not increased_themes.empty:
        most_increased_theme = increased_themes.iloc[0]
        most_increased_name = increased_themes.index[0]
    else:
        most_increased_theme = None
        most_increased_name = "无"

    # 创建报告
    report = f"""
    ============================
        节假日留言分析报告
    ============================

    总体统计:
    - 总留言数量: {total_messages:,} 条
    - 节假日期间留言: {holiday_messages:,} 条 ({holiday_percentage:.1f}%)
    - 非节假日留言: {total_messages - holiday_messages:,} 条 ({100 - holiday_percentage:.1f}%)

    节假日对比:
    - 节假日平均回复时长: {holiday_vs_normal.loc[True, '平均回复时长']:.1f} 天
    - 非节假日平均回复时长: {holiday_vs_normal.loc[False, '平均回复时长']:.1f} 天
    - 差异: {holiday_vs_normal.loc[True, '平均回复时长'] - holiday_vs_normal.loc[False, '平均回复时长']:.1f} 天

    最繁忙的节假日:
    - {busiest_holiday_name}: {int(busiest_holiday['留言数量'])} 条留言
    - 平均回复时长: {busiest_holiday['平均回复时长']:.1f} 天

    主题波动分析:
    - 节假日特殊主题数量: {len(increased_themes)}
    - 最显著的主题: {most_increased_name}
    - 特殊指数: {most_increased_theme['节假日特殊指数'] if most_increased_theme else 'N/A'}

    节假日前中后模式:
    """

    # 添加节假日前中后模式
    for context in context_counts.index:
        if '非节假日' not in str(context):
            count = int(context_counts.loc[context, '留言数量'])
            response_time = context_counts.loc[context, '平均回复时长']
            report += f"    - {context}: {count} 条留言，平均回复 {response_time:.1f} 天\n"

    # 保存报告
    with open('图像输出/节假日分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ 报告已保存至: 图像输出/节假日分析报告.txt")


def main():
    """主函数"""
    print("开始分析节假日期间的留言数量与主题波动...")

    # 1. 加载数据
    df = load_data()
    print(f"📊 加载数据完成，共 {len(df)} 条留言记录")

    # 2. 定义节假日期间
    holidays = define_holiday_periods()
    print(f"📅 定义 {len(holidays)} 个节假日期间")

    # 3. 添加节假日特征
    df = add_holiday_features(df, holidays)

    # 4. 分析节假日留言数量
    holiday_counts, holiday_vs_normal, context_counts = analyze_holiday_message_volume(df)

    # 5. 分析节假日主题分布
    theme_comparison, increased_themes = analyze_holiday_themes(df, top_n=15)

    # 6. 创建优化的主题对比图表（解决文字重叠问题）
    theme_counts_df = create_theme_comparison_chart(df, theme_comparison, holiday_vs_normal)

    # 7. 绘制主分析图表
    plot_holiday_message_analysis(df, holiday_counts, holiday_vs_normal, context_counts,
                                  theme_comparison, increased_themes)

    # 8. 创建分析报告
    create_holiday_summary_report(df, holiday_counts, holiday_vs_normal, context_counts,
                                  theme_comparison, increased_themes)

    # 9. 保存详细数据
    with pd.ExcelWriter('图像输出/节假日详细数据.xlsx', engine='openpyxl') as writer:
        holiday_counts.to_excel(writer, sheet_name='各节假日统计')
        holiday_vs_normal.to_excel(writer, sheet_name='节假日对比')
        context_counts.to_excel(writer, sheet_name='节假日前中后')
        theme_comparison.to_excel(writer, sheet_name='主题分布对比')
        increased_themes.to_excel(writer, sheet_name='显著增加主题')
        theme_counts_df.to_excel(writer, sheet_name='主题数量对比')

    print("\n✅ 分析完成！")
    print("📁 生成文件:")
    print("   - 图像输出/节假日主题数量对比_优化版.png (解决文字重叠问题)")
    print("   - 图像输出/节假日主题数量对比_简化版.png")
    print("   - 图像输出/节假日留言波动分析.png")
    print("   - 图像输出/节假日分析报告.txt")
    print("   - 图像输出/节假日详细数据.xlsx")

    print("\n📈 关键发现:")
    print(f"1. 节假日留言占比: {(len(df[df['is_holiday_period']]) / len(df) * 100):.1f}%")
    if not increased_themes.empty:
        print(f"2. 节假日期间显著增加的主题数: {len(increased_themes)}个")
        print(f"3. 最显著的节假日主题: {increased_themes.index[0]}")
    print(f"4. 最繁忙的节假日: {holiday_counts.index[0]}")


if __name__ == "__main__":
    main()