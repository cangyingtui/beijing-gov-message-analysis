import pandas as pd
import jieba
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from wordcloud import WordCloud
import networkx as nx
import random
import os

CUSTOM_COLORS = ['#1f3a89', '#528edc', '#a9d3e4', '#e19c64', '#f5d5a2']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("图像输出", exist_ok=True)
df = pd.read_pickle("processed_data.pkl")

# 自定义停用词
stopwords = set([
    '的','了','是','在','我','有','和','就','不','人','都','一','一个','上','也','很','到','说','去','你',
    '会','着','没有','看','好','自己','这','能',
    '要','可以','还','个','对','为','与','及','等','但',
    '或','而','您好','我们','是否','已经','谢谢','领导','希望','请问'
])

text = " ".join(df["content"].dropna().astype(str))
words = jieba.cut(text)
filtered = [w for w in words if len(w) > 1 and w not in stopwords]

word_counts = Counter(filtered)
top20 = word_counts.most_common(20)

# ===== 词频表 =====
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

table_data = []
for i in range(10):
    left = top20[i]
    right = top20[i+10] if i+10 < len(top20) else ("", "")
    table_data.append([left[0], left[1], right[0], right[1]])

table = ax.table(
    cellText=table_data,
    colLabels=["关键词", "词频", "关键词", "词频"],
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 3)

plt.title("10 Top20 高频词统计表", fontsize=18)
plt.savefig("图像输出/10_Top20词频表.png", dpi=300)
plt.close()

# ===== 词云 =====
def color_func(*args, **kwargs):
    return random.choice(CUSTOM_COLORS)

wordcloud = WordCloud(
    font_path="C:/Windows/Fonts/msyh.ttc",
    background_color="white",
    width=800,
    height=600,
    color_func=color_func
).generate(" ".join(filtered))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("11 留言内容词云图", fontsize=18)
plt.savefig("图像输出/11_留言词云图.png", dpi=300)
plt.close()

# ===== 共现网络 =====
top_words = [w for w, _ in top20]
cooccur = Counter()

for content in df["content"].dropna():
    ws = [w for w in jieba.cut(str(content)) if w in top_words]
    for w1, w2 in combinations(set(ws), 2):
        cooccur[tuple(sorted((w1, w2)))] += 1

G = nx.Graph()
for w in top_words:
    G.add_node(w, size=word_counts[w])

for (w1, w2), c in cooccur.items():
    if c >= 3:
        G.add_edge(w1, w2, weight=c)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=2)
nx.draw(
    G, pos,
    with_labels=True,
    node_size=[G.nodes[n]["size"] * 2 for n in G.nodes()],
    node_color="#b7dce8",
    edge_color="#8c8c8c",
    linewidths=0.5,
    alpha=0.9,
    font_family="Microsoft YaHei",
    font_size=10
)

plt.title("12 高频词语义共现网络", fontsize=18)
plt.savefig("图像输出/12_共现网络图.png", dpi=300)
plt.close()
