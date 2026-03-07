# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import geopandas as gpd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, no_update
import chinese_calendar as cc

# =========================
# 0) 常量
# =========================
DATA_PATH = "processed_data_with_district.pkl"
GEO_PATH = "北京市_县.geojson"

BJ_DISTRICTS = [
    "东城区","西城区","朝阳区","海淀区","丰台区","石景山区",
    "门头沟区","房山区","通州区","顺义区","昌平区","大兴区",
    "怀柔区","平谷区","密云区","延庆区"
]

# 你的主题 5 色（全站主色）
BASE_COLORS = [
    "rgb(32,56,136)",   # 深蓝
    "rgb(81,141,219)",  # 蓝
    "rgb(167,210,228)", # 浅蓝
    "rgb(245,215,163)", # 浅橙
    "rgb(225,156,102)"  # 橙
]

# =========================
# 1) 邻近插补扩展色板（给 Top10/饼图循环用）
# =========================
def _rgb_to_tuple(s: str):
    s = s.strip().lower().replace("rgb(", "").replace(")", "")
    r, g, b = [int(x) for x in s.split(",")]
    return r, g, b

def _tuple_to_rgb(t):
    return f"rgb({t[0]},{t[1]},{t[2]})"

def _lerp(a, b, t):
    return int(round(a + (b - a) * t))

def expand_palette(base_colors, n=12):
    base = [_rgb_to_tuple(c) for c in base_colors]
    if n <= len(base_colors):
        return base_colors[:n]

    seg = len(base) - 1
    need = n - len(base)
    per = [need // seg] * seg
    for i in range(need % seg):
        per[i] += 1

    out = [base[0]]
    for i in range(seg):
        a, b = base[i], base[i + 1]
        k = per[i]
        for j in range(1, k + 1):
            t = j / (k + 1)
            out.append((_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t)))
        out.append(b)

    out = out[:n]
    return [_tuple_to_rgb(x) for x in out]

DISCRETE_COLORS = expand_palette(BASE_COLORS, n=12)  # ✅ Top10/饼图循环色
CONTINUOUS_SCALE = BASE_COLORS                       # ✅ 地图连续渐变色

# =========================
# 2) 数据加载与预处理
# =========================
df = pd.read_pickle(DATA_PATH)
df = df[df["district_final"].isin(BJ_DISTRICTS)].copy()
df["message_time"] = pd.to_datetime(df["message_time"], errors="coerce")
df["is_overtime"] = df["duration_days"].apply(lambda x: x > 15 if pd.notnull(x) else False)

summary = df.groupby("district_final").agg(
    total_messages=("message_id", "count"),
    avg_duration=("duration_days", "mean"),
    percent_overtime=("is_overtime", lambda x: round(100 * x.sum() / x.count(), 1))
).reset_index().rename(columns={"district_final": "district"})

# =========================
# 3) 地图数据加载与合并
# =========================
gdf = gpd.read_file(GEO_PATH)

# 区名字段统一
if "district" not in gdf.columns:
    if "name" in gdf.columns:
        gdf = gdf.rename(columns={"name": "district"})
    else:
        raise ValueError(f"GeoJSON 找不到 district/name 字段：{list(gdf.columns)}")

gdf["district"] = gdf["district"].astype(str).str.strip()
summary["district"] = summary["district"].astype(str).str.strip()

gdf = gdf[gdf["district"].isin(BJ_DISTRICTS)].copy()

# MultiPolygon 取第一个 Polygon（避免 choropleth_mapbox 兼容性问题）
gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: geom if geom.geom_type == "Polygon" else list(geom.geoms)[0]
)

gdf_map = gdf.merge(summary, on="district", how="left")
gdf_map["total_messages"] = gdf_map["total_messages"].fillna(0)
gdf_map["avg_duration"] = gdf_map["avg_duration"].fillna(0)
gdf_map["percent_overtime"] = gdf_map["percent_overtime"].fillna(0)

# ✅ 计算 bounds：用于“地图完整显示”（替代 fitbounds）
minx, miny, maxx, maxy = gdf_map.total_bounds
CENTER_LON = float((minx + maxx) / 2)
CENTER_LAT = float((miny + maxy) / 2)

# =========================
# 4) 节假日信息
# =========================
def get_date_info(date_obj: datetime.date):
    detail = cc.get_holiday_detail(date_obj)
    holiday_name = "非特殊节假日"
    if detail:
        if isinstance(detail, (list, tuple)) and len(detail) > 0 and detail[0]:
            holiday_name = str(detail[0])
        elif isinstance(detail, str):
            holiday_name = detail
    is_workday = cc.is_workday(date_obj)
    return is_workday, holiday_name

# =========================
# 5) 图表生成（全站复用）
# =========================
def make_unit_pie(sub_df, title):
    unit_count = sub_df["reply_unit"].fillna("未知单位").value_counts().nlargest(10)
    fig = px.pie(
        values=unit_count.values,
        names=unit_count.index,
        title=title,
        color_discrete_sequence=DISCRETE_COLORS  # ✅ 循环色
    )
    return fig

def make_efficiency_bar(sub_df, title):
    avg_days = float(sub_df["duration_days"].mean()) if sub_df["duration_days"].notna().any() else 0.0
    overtime_pct = float(sub_df["is_overtime"].mean() * 100) if len(sub_df) else 0.0
    fig = px.bar(
        x=["平均处理时长（天）", "超时率（%）"],
        y=[round(avg_days, 2), round(overtime_pct, 2)],
        title=title,
        labels={"x": "指标", "y": "值"},
        color_discrete_sequence=[DISCRETE_COLORS[0]]
    )
    fig.update_layout(yaxis_title="")
    return fig

def make_top10_questions(sub_df, title):
    top10 = sub_df["title"].fillna("（无标题）").value_counts().nlargest(10)
    fig = px.bar(
        x=top10.values[::-1],
        y=top10.index[::-1],
        orientation="h",
        title=title,
        labels={"x": "留言数量", "y": "问题标题"}
    )
    # ✅ 每条 bar 单独上色（循环）
    colors = (DISCRETE_COLORS * 3)[:len(top10)]
    fig.update_traces(marker=dict(color=colors[::-1]))
    fig.update_layout(margin={"l": 140, "r": 20, "t": 60, "b": 30})
    return fig

def make_city_map_fig():
    # ✅ 关键：用 locations=gdf_map.index，clickData['location'] 就是 index（你的第一段联动就是这么做的）
    fig = px.choropleth_mapbox(
        gdf_map,
        geojson=gdf_map.geometry.__geo_interface__,
        locations=gdf_map.index,
        color="total_messages",
        hover_name="district",
        hover_data={"total_messages": True, "avg_duration": True, "percent_overtime": True},
        mapbox_style="carto-positron",
        center={"lat": CENTER_LAT, "lon": CENTER_LON},
        zoom=8,
        opacity=0.65,
        color_continuous_scale=CONTINUOUS_SCALE
    )
    # ✅ 用 bounds 保证完整显示（不使用 fitbounds，避免你本机 plotly 报错）
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox=dict(
            bounds={"west": float(minx), "south": float(miny), "east": float(maxx), "north": float(maxy)}
        )
    )
    return fig

CITY_MAP_FIG = make_city_map_fig()

# =========================
# 6) Dash 初始化（含二次跳转）
# =========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "北京留言分析仪表盘"

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="user-selection", storage_type="session"),       # 主页提交 → dashboard
    dcc.Store(id="city-selected", storage_type="session", data=None),  # city 点击地图选区

    dcc.Markdown("""
    <style>
    @keyframes blink { 0%{opacity:1;} 50%{opacity:0.2;} 100%{opacity:1;} }

    body { margin: 0; font-size: 16px; }
    h1 { font-size: 30px; }
    h2 { font-size: 26px; }
    h3 { font-size: 22px; }
    label { font-size: 16px; font-weight: 600; }
    a { font-size: 14px; }

    .card {
      border: 1px solid #eee;
      border-radius: 12px;
      padding: 16px;
      background: #fff;
      box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .section-title { font-size: 18px; font-weight: 700; margin: 10px 0; }

    </style>
    """, dangerously_allow_html=True),

    html.Div(id="page-content")
])

# -------------------------
# 三页 layout
# -------------------------
home_layout = html.Div([
    html.H2("📍 北京留言智能分析平台", style={"textAlign": "center", "marginTop": "16px"}),

    html.Div([
        html.Label("请选择城区："),
        dcc.Dropdown(BJ_DISTRICTS, id="input-district", value="朝阳区", clearable=False),

        html.Br(),
        html.Label("请选择日期："),
        dcc.DatePickerSingle(id="input-date", date=datetime.date.today(), display_format="YYYY-MM-DD"),

        html.Br(), html.Br(),
        html.Label("请选择咨询类型（模拟字段）："),
        dcc.Dropdown(["物业管理","交通出行","民生问题","社会治安"], id="input-topic", value="物业管理", clearable=False),

        html.Br(),
        html.Button("提交", id="submit-btn", n_clicks=0, style={"width":"100%","height":"42px"})
    ], style={
        "width":"420px","margin":"18px auto","padding":"16px",
        "border":"1px solid #eee","borderRadius":"12px"
    })
])

dashboard_layout = html.Div([
    # ✅ 二次跳转入口：/dashboard -> /city
    html.A("▶ 点击此查看全市地图 ⮕", href="/city", style={
        "color": "red", "fontWeight": "bold", "fontSize": "22px",
        "textDecoration": "none", "animation": "blink 1s infinite",
        "position": "absolute", "top": "18px", "right": "26px", "zIndex": 9999
    }),

    html.Div(className="nav", style={"padding":"12px 18px"}, children=[
        html.A("← 返回主页", href="/")
    ]),

    html.H3("📌 单区仪表盘", style={"textAlign":"center","marginTop":"6px"}),
    html.Div(id="user-output", style={"margin":"10px 18px","textAlign":"center","fontSize":"16px"}),

    html.Div([
        dcc.Graph(id="dash-unit-pie", style={"display":"inline-block","width":"49%"}),
        dcc.Graph(id="dash-efficiency", style={"display":"inline-block","width":"49%"})
    ], style={"width":"96%","margin":"0 auto"}),

    html.H3("📌 高频咨询问题 Top 10（按标题统计）", style={"textAlign":"center","marginTop":"10px"}),
    dcc.Graph(id="dash-top10", style={"width":"96%","margin":"0 auto"})
], style={"position":"relative"})

city_layout = html.Div([
    html.Div(className="nav", style={"padding":"12px 18px"}, children=[
        html.A("← 返回单区仪表盘", href="/dashboard"),
        html.A("← 返回主页", href="/")
    ]),

    html.H2("📍 北京市各区留言分析仪表盘（可长按鼠标拖拽地图）", style={"textAlign":"center","margin":"6px 0"}),
    html.P("点击地图城区 → 下方三图联动更新（Top10/回复单位/处理效率）。",
           style={"textAlign":"center","margin":"0 0 10px 0"}),

    # ✅ 地图预置 figure：不靠回调重画（更稳），但 clickData 仍然正常触发
    dcc.Graph(id="map-graph", figure=CITY_MAP_FIG, style={"height":"78vh"}, config={"responsive": True}),

    html.Div(id="city-info", style={"textAlign":"center","fontSize":"16px","marginTop":"10px"}),

    html.Div([
        dcc.Graph(id="city-unit-pie", style={"display":"inline-block","width":"49%"}),
        dcc.Graph(id="city-efficiency", style={"display":"inline-block","width":"49%"})
    ], style={"width":"96%","margin":"0 auto"}),

    html.H3("📌 高频咨询问题 Top 10（按标题统计）", style={"textAlign":"center","marginTop":"10px"}),
    dcc.Graph(id="city-top10", style={"width":"96%","margin":"0 auto"})
])

# 防白屏
app.validation_layout = html.Div([app.layout, home_layout, dashboard_layout, city_layout])

# =========================
# 7) 路由
# =========================
@app.callback(Output("page-content","children"), Input("url","pathname"))
def route(pathname):
    if pathname == "/dashboard":
        return dashboard_layout
    if pathname == "/city":
        return city_layout
    return home_layout

# =========================
# 8) 主页提交 -> /dashboard
# =========================
@app.callback(
    Output("user-selection","data"),
    Output("url","pathname"),
    Input("submit-btn","n_clicks"),
    State("input-district","value"),
    State("input-date","date"),
    State("input-topic","value"),
    prevent_initial_call=True
)
def submit_and_route(n, district, date, topic):
    payload = {"district": district, "date": date, "topic": topic}
    return payload, "/dashboard"

# =========================
# 9) /dashboard 图表渲染
# =========================
@app.callback(
    Output("user-output","children"),
    Output("dash-unit-pie","figure"),
    Output("dash-efficiency","figure"),
    Output("dash-top10","figure"),
    Input("user-selection","data"),
    Input("url","pathname")
)
def update_dashboard(selection, pathname):
    if pathname != "/dashboard":
        return no_update, no_update, no_update, no_update

    if not selection:
        empty = px.bar(title="请先返回主页选择条件并提交")
        return "⚠ 未检测到输入条件，请返回主页重新提交。", empty, empty, empty

    district = selection.get("district", "朝阳区")
    date = selection.get("date", datetime.date.today().strftime("%Y-%m-%d"))
    topic = selection.get("topic", "物业管理")

    try:
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        date_obj = datetime.date.today()

    is_workday, holiday_name = get_date_info(date_obj)
    day_type = "工作日" if is_workday else "节假日"

    sub_df = df[df["district_final"] == district].copy()
    if sub_df.empty:
        empty = px.bar(title="当前区域暂无数据")
        return "⚠ 当前区域暂无数据", empty, empty, empty

    desc = (f"当前为【{district}】的【{topic}】咨询。日期：{date_obj}，属于【{day_type}】（节假日类型：{holiday_name}）。"
            f"过往平均处理时长 {sub_df['duration_days'].mean():.1f} 天；过往超时率 {sub_df['is_overtime'].mean()*100:.1f}%。")

    return (
        desc,
        make_unit_pie(sub_df, f"{district}｜回复单位分布（Top10）"),
        make_efficiency_bar(sub_df, f"{district}｜处理效率指标"),
        make_top10_questions(sub_df, f"{district}｜高频问题 Top10（标题统计）")
    )

# =========================
# 10) /city：点击地图 -> 选中区（完全照你第一段“动态很好”的逻辑）
# =========================
@app.callback(
    Output("city-selected","data"),
    Input("map-graph","clickData"),
    State("city-selected","data"),
    prevent_initial_call=True
)
def pick_district(clickData, current):
    if clickData and "points" in clickData and len(clickData["points"]) > 0:
        idx = clickData["points"][0].get("location", None)
        if idx is not None and idx in gdf_map.index:
            return gdf_map.loc[idx, "district"]
    return current  # 不点就保持上一次

# =========================
# 11) /city：根据选中区联动三图（不消失）
# =========================
@app.callback(
    Output("city-info","children"),
    Output("city-unit-pie","figure"),
    Output("city-efficiency","figure"),
    Output("city-top10","figure"),
    Input("city-selected","data"),
    Input("url","pathname")
)
def update_city(selected, pathname):
    if pathname != "/city":
        return no_update, no_update, no_update, no_update

    # 默认：不点则展示全市
    if not selected:
        sub_df = df.copy()
        label = "全市"
        info = "📌 当前选中区域：全市（点击地图可切换单区）"
    else:
        sub_df = df[df["district_final"] == selected].copy()
        label = selected
        info = f"📌 当前选中区域：{selected}"

    if sub_df.empty:
        empty = px.bar(title="暂无数据")
        return info + "（暂无数据）", empty, empty, empty

    return (
        info,
        make_unit_pie(sub_df, f"{label}｜回复单位分布（Top10）"),
        make_efficiency_bar(sub_df, f"{label}｜处理效率指标"),
        make_top10_questions(sub_df, f"{label}｜高频问题 Top10（标题统计）")
    )

# =========================
# 12) 启动
# =========================
if __name__ == "__main__":
    app.run(debug=True)
