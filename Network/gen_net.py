import networkx as nx
import pandas as pd
import community as community_louvain
from collections import defaultdict
import matplotlib.pyplot as plt

import os

INPUT_FILE = r"WorkData\matrix\2122_matrix_article.csv"
OUTPUT_PATH = r"WorkData\network\2122"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
df_matrix = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
print(df_matrix.head(10))

def generate_undirected_graph(df_matrix):
    # 创建无向图
    G = nx.Graph()
    for _, row in df_matrix.iterrows():
        G.add_edge(row["Entity1"], row["Entity2"], weight=row["CoOccurrence"])

    # 输出图的基本信息
    print("\n网络信息：")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")

    return G
def generate_directed(G):
    degree_centrality = nx.degree_centrality(G)
    weighted_degree = {node: sum(d["weight"] for _, d in G[node].items()) for node in G.nodes}
    # 保存到 CSV 文件
    df_metrics = pd.DataFrame({
        "Entity": degree_centrality.keys(),
        "DegreeCentrality": degree_centrality.values(),  # 节点连接数
        "WeightedDegree": weighted_degree.values()       # 加权度中心性
    })
    df_metrics.to_csv(f"{OUTPUT_PATH}\\node_metrics.csv", encoding='utf-8-sig', index=False)

    print("\n节点重要性指标已保存到 node_metrics.csv")

def detect_communities(G):
    """
    使用Louvain算法进行社区检测
    参数:
        G: 网络图对象
    返回:
        dict: 节点到社区ID的映射
    """
    # 需确保networkx版本>=2.0，权重参数正确传递
    partition = community_louvain.best_partition(G, weight='weight')
    
    # 转换为社区ID到节点列表的映射
    community_map = defaultdict(list)
    for node, community_id in partition.items():
        community_map[community_id].append(node)
    
    # 保存社区结果
    communities_df = pd.DataFrame([
        {'CommunityID': cid, 'Entities': ', '.join(entities)} 
        for cid, entities in community_map.items()
    ])
    communities_df.to_csv(f'{OUTPUT_PATH}\\communities.csv',encoding='utf-8-sig', index=False)
    print("\n社区检测结果已保存到 communities.csv")
    return partition

def generate_directed_graph(G, partition):
    # 设置画布大小
    plt.figure(figsize=(12, 8))

    # 使用 spring_layout 布局（模拟物理系统）
    pos = nx.spring_layout(G, k=0.5)  # k 控制节点间距

    # 设置节点颜色（按社区）
    community_colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFD700"]  # 颜色列表
    node_colors = [community_colors[partition[node] % len(community_colors)] for node in G.nodes]

    # 绘制网络
    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=[v * 10 for v in nx.degree(G)],  # 节点大小 = 度数 × 10
            width=[d["weight"] * 0.5 for _, _, d in G.edges(data=True)],  # 边粗细 = 权重 × 0.5
            alpha=0.8,
            font_size=10,
            edge_color="gray")

    # 添加标题和保存图像
    plt.title("Industry-Technology-Capital Network")
    plt.savefig("network_graph.png", dpi=300, bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
    pass
    # G = generate_undirected_graph(df_matrix)
    # generate_directed_graph(G)
    # partition = detect_communities(G)
    # generate_directed_graph(G, partition)
    