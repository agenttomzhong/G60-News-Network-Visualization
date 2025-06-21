import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from gen_net import generate_undirected_graph, detect_communities
import os
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors
import math
import networkx as nx
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotly.colors import qualitative
from plotly.colors import hex_to_rgb


def read_matrix_article():
    """Read matrix file and return a pandas DataFrame"""
    df = {}
    years = ['2122', '2223', '2324', '2425']
    for year in years:
        file_path = f'data/{year}_matrix_article.csv'
        df[year] = pd.read_csv(file_path)
    return df

def read_entity_dict():
    df = pd.read_csv(r"data\entity_dict.csv")
    return df

def read_word_freq(file_path):
    """Read word frequency file and return a pandas DataFrame"""
    df = {}
    years = ['2122', '2223', '2324', '2425']
    for year in years:
        file_path = f'data/{year}_t_f.csv'
        df[year] = pd.read_csv(file_path)

    return df

# def load_map_json():


def load_map_location():
    changjiang_delta_places = {
        '上海': [['上海', '上海市'], 121.47, 31.23],
        '江苏': [['江苏', '江苏省'], 118.78, 32.07],
        '浙江': [['浙江', '浙江省'], 120.15, 30.28],
        '安徽': [['安徽', '安徽省'], 117.28, 31.86],
        '南京': [['南京', '南京市'], 118.78, 32.07],
        '苏州': [['苏州', '苏州市'], 120.62, 31.32],
        '杭州': [['杭州', '杭州市'], 120.15, 30.28],
        '合肥': [['合肥', '合肥市'], 117.28, 31.86],
        '宁波': [['宁波', '宁波市'], 121.55, 29.88],
        '无锡': [['无锡', '无锡市'], 120.30, 31.57],
        '常州': [['常州', '常州市'], 119.95, 31.78],
        '南通': [['南通', '南通市'], 120.86, 32.01],
        '扬州': [['扬州', '扬州市'], 119.42, 32.39],
        '镇江': [['镇江', '镇江市'], 119.45, 32.20],
        '泰州': [['泰州', '泰州市'], 119.90, 32.49],
        '盐城': [['盐城', '盐城市'], 120.15, 33.35],
        '淮安': [['淮安', '淮安市'], 119.02, 33.62],
        '宿迁': [['宿迁', '宿迁市'], 118.28, 33.96],
        '徐州': [['徐州', '徐州市'], 117.18, 34.26],
        '连云港': [['连云港', '连云港市'], 119.16, 34.59],
        '温州': [['温州', '温州市'], 120.70, 28.00],
        '绍兴': [['绍兴', '绍兴市'], 120.58, 30.01],
        '湖州': [['湖州', '湖州市'], 120.09, 30.89],
        '嘉兴': [['嘉兴', '嘉兴市'], 120.76, 30.77],
        '金华': [['金华', '金华市'], 119.65, 29.08],
        '衢州': [['衢州', '衢州市'], 118.87, 28.94],
        '台州': [['台州', '台州市'], 121.42, 28.66],
        '丽水': [['丽水', '丽水市'], 119.92, 28.45],
        '舟山': [['舟山', '舟山市'], 122.20, 30.00],
        '芜湖': [['芜湖', '芜湖市'], 118.38, 31.33],
        '马鞍山': [['马鞍山', '马鞍山市'], 118.51, 31.70],
        '铜陵': [['铜陵', '铜陵市'], 117.82, 30.94],
        '安庆': [['安庆', '安庆市'], 117.07, 30.52],
        '滁州': [['滁州', '滁州市'], 118.32, 32.30],
        '池州': [['池州', '池州市'], 117.49, 30.67],
        '宣城': [['宣城', '宣城市'], 118.76, 30.95],
        '松江': [['松江', '松江区'], 121.23, 31.03],
        '虹桥': [['虹桥'], 121.38, 31.20],
        '张江': [['张江'], 121.58, 31.20],
        '临港': [['临港'], 121.88, 30.90],
        '昆山': [['昆山', '昆山市'], 120.95, 31.39],
        '嘉善': [['嘉善', '嘉善县'], 120.92, 30.84],
        '嘉兴港区': [['嘉兴港区'], 121.10, 30.60],
        '杭州湾': [['杭州湾'], 121.15, 30.30],
        '金华经开区': [['金华经开区'], 119.65, 29.08],
        '合肥高新区': [['合肥高新区'], 117.20, 31.80],
        '苏州工业园': [['苏州工业园', '苏州工业园区'], 120.70, 31.32]
    }
    return changjiang_delta_places


def create_sankey_diagram():
    """Create animated Sankey diagram showing co-occurrence of entities over years"""
    dataframes = read_matrix_article()  # Using the centralized data loading function

    merged_data = pd.concat([df.assign(year=year) for year, df in dataframes.items()], ignore_index=True)
    unique_entities = list(set(merged_data['Entity1']).union(set(merged_data['Entity2'])))
    entity_to_index = {entity: idx for idx, entity in enumerate(unique_entities)}
    frames = []

    for year in dataframes.keys():  # Use the years from the loaded data
        yearly_data = merged_data[merged_data['year'] == year]
        source_indices = []
        target_indices = []
        values = []

        for index, row in yearly_data.iterrows():
            source_idx = entity_to_index[row['Entity1']]
            target_idx = entity_to_index[row['Entity2']]
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['CoOccurrence'])

        frame_data = go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=unique_entities
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )
        frames.append(go.Frame(data=[frame_data], name=str(year)))

    layout = go.Layout(
        title='Co-occurrence of Entities Over Four Years',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None,
                               {'fromcurrent': True, 'transition': {'duration': 500}, 'frame': {'duration': 1000}}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate', 'transition': {'duration': 0}}])
                ]
            )
        ],
        sliders=[
            dict(
                currentvalue={'prefix': 'Year: '},
                steps=[
                    dict(method='animate',
                         args=[[str(year)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                         label=str(year)) for year in dataframes.keys()  # Use loaded years
                ]
            )
        ]
    )

    initial_source_indices = []
    initial_target_indices = []
    initial_values = []

    first_year = list(dataframes.keys())[0]  # Get first year from loaded data
    for index, row in merged_data[merged_data['year'] == first_year].iterrows():
        initial_source_indices.append(entity_to_index[row['Entity1']])
        initial_target_indices.append(entity_to_index[row['Entity2']])
        initial_values.append(row['CoOccurrence'])

    initial_frame_data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=unique_entities
        ),
        link=dict(
            source=initial_source_indices,
            target=initial_target_indices,
            value=initial_values
        )
    )

    fig = go.Figure(data=[initial_frame_data], layout=layout, frames=frames)
    return fig


def create_parallel_bar_chart():
    """Create parallel bar chart showing community detection in relationship networks"""
    dataframes = read_matrix_article()  # Using the centralized data loading function
    partitions = {}

    for year, df in dataframes.items():
        G = generate_undirected_graph(df)
        partitions[year] = detect_communities(G)

    all_community_ids = set()
    for partition in partitions.values():
        all_community_ids.update(partition.values())
    community_ids = list(all_community_ids)
    num_communities = len(community_ids)

    if num_communities < 3:
        for i in range(3 - num_communities):
            community_ids.append(max(community_ids) + 1)

    colors = ['#FFFF99', '#ADD8E6', '#90EE90']
    nodes = []

    for year, data in dataframes.items():
        partition = partitions[year]

        for entity in data['Entity1'].unique():
            total_cooccurrence = data[(data['Entity1'] == entity) | (data['Entity2'] == entity)]['CoOccurrence'].sum()
            size = min(4 + total_cooccurrence * 0.4, 10)
            community_color = partition.get(entity, -1)
            color = colors[community_color % 3]

            nodes.append({
                'entity': entity,
                'year': year,
                'size': size,
                'color': color,
                'community': community_color
            })

    fig = go.Figure()

    for community_id, color in zip(range(len(colors)), colors):
        community_nodes = [node for node in nodes if node['community'] == community_id]
        entities = [node['entity'] for node in community_nodes]
        sizes = [node['size'] for node in community_nodes]
        years = [node['year'] for node in community_nodes]

        fig.add_trace(go.Bar(
            x=years,
            y=sizes,
            name=f'Community {community_id}',
            marker=dict(color=color),
            customdata=entities,
            hovertemplate='Year: %{x}<br>Entity: %{customdata}<br>Size: %{y}'
        ))

    fig.update_layout(
        title='bar chart',
        xaxis_title='year',
        yaxis_title='共现次数',
        barmode='group',
        width=1000,
        height=800,
        margin=dict(l=50, r=50, b=50, t=100),
        paper_bgcolor='white'
    )
    return fig


def create_line_chart():
    """Create line chart showing G60 word frequency changes"""
    changjiang_delta_places = load_map_location()
    file_paths = [
        r"data\2122_t_f.csv",
        r"data\2223_t_f.csv",
        r"data\2324_t_f.csv",
        r"data\2425_t_f.csv"
    ]

    dfs = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            year = os.path.basename(file_path)[:4]
            df['year'] = year
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    place_freq = {}

    for _, row in combined_df.iterrows():
        word = row['word']
        freq = row['frequency']
        year = row['year']

        for place_name, (variants, lon, lat) in changjiang_delta_places.items():
            if word in variants:
                if place_name not in place_freq:
                    place_freq[place_name] = {}
                if year not in place_freq[place_name]:
                    place_freq[place_name][year] = 0
                place_freq[place_name][year] += freq
                break

    result_df_list = []
    for place_name, yearly_freq in place_freq.items():
        for year, freq in yearly_freq.items():
            result_df_list.append({'place': place_name, 'year': year, 'frequency': freq})

    place_freq_df = pd.DataFrame(result_df_list).sort_values(by=['place', 'year'])
    fig = px.line(place_freq_df, x='year', y='frequency', color='place',
                  title='G60 word frequncy change (2021-2025)',
                  labels={'year': '年份', 'frequency': '词频', 'place': '地名'},
                  markers=True)

    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        yaxis_title='词频',
        legend_title_text='地名'
    )
    return fig


def create_cooccurrence_network():
    """Create co-occurrence network visualization"""
    df = read_entity_dict()

    category_colors = {
        '产业': to_hex(mcolors.to_rgb('salmon')),
        '技术': to_hex(mcolors.to_rgb('dodgerblue')),
        '资本': to_hex(mcolors.to_rgb('mediumseagreen'))
    }

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['entity'],
                   category=row['category'],
                   frequency=row['frequency'],
                   reason=row['reason'],
                   size=math.log(row['frequency']) * 3 + 10,
                   color=category_colors[row['category']])

    for i in range(len(df) - 1):
        if df.iloc[i]['category'] == df.iloc[i + 1]['category']:
            G.add_edge(df.iloc[i]['entity'], df.iloc[i + 1]['entity'], weight=0.5)

    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(136, 136, 136, 0.5)'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_info = f"<b>{node}</b><br>Category: {G.nodes[node]['category']}<br>Frequency: {G.nodes[node]['frequency']}<br>Description: {G.nodes[node]['reason']}"
        node_text.append(node_info)
        node_size.append(G.nodes[node]['size'])
        node_color.append(G.nodes[node]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='<b>Co-occurrence Network Analysis</b><br><span style="font-size:12px">产业(橙) | 技术(蓝) | 资本(绿)</span>',
                            font=dict(size=24, family='Arial')
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=60),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_x=0.5,
        legend_title_text='<b>Categories</b>',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=12),
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal'
        ),
        width=1200,
        height=800,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    return fig




def create_force_directed_network():
    """Create force-directed network visualization matching the matplotlib version"""
    years = ['2122', '2223', '2324', '2425']
    sample_data = {
        '2122': [('发展', 8676), ('长三角', 7189), ('企业', 6240), ('产业', 5868), ('创新', 5494)],
        '2223': [('发展', 8000), ('长三角', 7000), ('科创', 6500), ('产业', 6000), ('G60', 5500)],
        '2324': [('科创', 7500), ('G60', 7000), ('发展', 6800), ('长三角', 6500), ('一体化', 6000)],
        '2425': [('一体化', 8000), ('科创', 7800), ('G60', 7500), ('数字经济', 7000), ('发展', 6800)]
    }

    # Create graph
    G = nx.DiGraph()
    colors = qualitative.Plotly[:len(years)]  # Using Plotly's color sequence

    for i, year in enumerate(years):
        for word, freq in sample_data[year]:
            node_id = f"{word}_{year}"
            G.add_node(node_id,
                       year=year,
                       word=word,
                       frequency=freq,
                       color=colors[i])

    # Add edges
    for i in range(len(years) - 1):
        current_year = years[i]
        next_year = years[i + 1]

        # Connect same words across years (solid lines)
        common_words = set([w for w, _ in sample_data[current_year]]).intersection(
            set([w for w, _ in sample_data[next_year]]))
        for word in common_words:
            G.add_edge(f"{word}_{current_year}", f"{word}_{next_year}",
                       width=1.5, color='#1f78b4', dash='solid')

        # Connect same rank words (dashed lines)
        min_len = min(len(sample_data[current_year]), len(sample_data[next_year]))
        for rank in range(min_len):
            current_word = sample_data[current_year][rank][0]
            next_word = sample_data[next_year][rank][0]
            if current_word != next_word:
                G.add_edge(f"{current_word}_{current_year}", f"{next_word}_{next_year}",
                           width=1.0, color='#a6cee3', dash='dash')

    # Use the exact same manual positions as in the matplotlib version
    manual_positions = {
        "发展_2122": (0, 2),
        "长三角_2122": (0, 1),
        "企业_2122": (0, 0),
        "产业_2122": (0, -1),
        "创新_2122": (0, -2),

        "发展_2223": (3, 2.2),
        "长三角_2223": (3, 1),
        "科创_2223": (3, 0),
        "产业_2223": (3, -1),
        "G60_2223": (3, -2),

        "科创_2324": (6, 2),
        "G60_2324": (6, 1),
        "发展_2324": (6, 0),
        "长三角_2324": (6, -1),
        "一体化_2324": (6, -2),

        "一体化_2425": (9, 2),
        "科创_2425": (9, 1),
        "G60_2425": (9, 0),
        "数字经济_2425": (9, -1),
        "发展_2425": (9, -2)
    }

    # Apply manual positions
    pos = {}
    for node in G.nodes():
        if node in manual_positions:
            pos[node] = manual_positions[node]
        else:
            year = G.nodes[node]['year']
            year_idx = years.index(year)
            pos[node] = (year_idx * 3, 0)  # Default position if not specified

    # Create Plotly figure
    edge_traces = []

    # Separate solid and dashed edges for styling
    solid_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['dash'] == 'solid']
    dashed_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['dash'] == 'dash']

    # Add solid edges
    for edge in solid_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1.5, color='#1f78b4'),
            hoverinfo='none',
            mode='lines',
            showlegend=False)
        edge_traces.append(edge_trace)

    # Add dashed edges
    for edge in dashed_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1.0, color='#a6cee3', dash='dash'),
            hoverinfo='none',
            mode='lines',
            showlegend=False)
        edge_traces.append(edge_trace)

    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['word'])
        node_hovertext.append(
            f"{G.nodes[node]['word']} ({G.nodes[node]['year']})<br>Frequency: {G.nodes[node]['frequency']}")
        node_color.append(G.nodes[node]['color'])
        node_size.append(10 + math.log(G.nodes[node]['frequency']) * 2)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hovertext=node_hovertext,
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=0.8, color='white')),
        textfont=dict(
            family="SimHei",
            size=12,
            color="black"
        )
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='Force-Directed Network of Word Evolution',
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 11]),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 4]),
                        width=1000,
                        height=600,
                        plot_bgcolor='white'))

    # Add year annotations to match matplotlib version
    year_annotations = [
        dict(x=0, y=3.5, text="2021-2022年", showarrow=False, font=dict(size=12)),
        dict(x=3, y=3.5, text="2022-2023年", showarrow=False, font=dict(size=12)),
        dict(x=6, y=3.5, text="2023-2024年", showarrow=False, font=dict(size=12)),
        dict(x=9, y=3.5, text="2024-2025年", showarrow=False, font=dict(size=12))
    ]

    # Add legend for edge types
    fig.update_layout(
        annotations=year_annotations,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='white',
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    # Add custom legend items for edge types
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(width=1.5, color='#1f78b4'),
        name='same word'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(width=1.0, color='#a6cee3', dash='dash'),
        name='row'
    ))

    return fig


def create_heatmap():
    """Create co-occurrence heatmap visualization"""
    dataframes = read_matrix_article()

    merged_data = pd.concat([df.assign(year=year) for year, df in dataframes.items()], ignore_index=True)
    pivot_table = merged_data.pivot_table(index='Entity1', columns='Entity2', values='CoOccurrence', aggfunc='sum',
                                          fill_value=0)
    pivot_table.sort_index(inplace=True)
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    fig = px.imshow(pivot_table,
                    labels=dict(x="Entity2", y="Entity1", color="Co-occurrence"),
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    color_continuous_scale='YlGnBu')

    fig.update_layout(
        title='Four-Year Co-occurrence Matrix Heatmap',
        xaxis_title='Entity2',
        yaxis_title='Entity1',
        width=1000,
        height=800,
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed")
    )
    return fig



def create_changjiang_map():
    """Create Changjiang Delta map visualization"""
    changjiang_delta_places = load_map_location()

    file_paths = [
        r"data\2122_t_f.csv",
        r"data\2223_t_f.csv",
        r"data\2324_t_f.csv",
        r"data\2425_t_f.csv"
    ]

    combined_df = pd.concat([pd.read_csv(fp) for fp in file_paths], ignore_index=True)

    place_freq = {}
    for place_name, (variants, lon, lat) in changjiang_delta_places.items():
        place_freq[place_name] = {
            'frequency': combined_df[combined_df['word'].isin(variants)]['frequency'].sum(),
            'longitude': lon,
            'latitude': lat
        }

    place_freq_df = pd.DataFrame.from_dict(place_freq, orient='index')
    place_freq_df = place_freq_df[place_freq_df['frequency'] > 0]
    place_freq_df.reset_index(inplace=True)
    place_freq_df.rename(columns={'index': 'place'}, inplace=True)

    fig = px.scatter_geo(place_freq_df,
                         lat='latitude',
                         lon='longitude',
                         size='frequency',
                         hover_name='place',
                         hover_data={'frequency': True, 'latitude': False, 'longitude': False},
                         scope='asia',
                         title='G60 Word Frequency Map (2021-2025)')

    fig.update_geos(
        resolution=50,
        showcountries=True,
        showsubunits=True,
        landcolor='lightgray',
        oceancolor='lightblue'
    )

    fig.update_layout(
        geo=dict(
            center=dict(lat=30, lon=120),
            projection_scale=5
        ),
        width=1000,
        height=800
    )

    return fig


def create_shanghai_map():
    """Create Shanghai map visualization"""
    shanghai_places = {
        '上海': [['上海', '上海市'], 121.47, 31.23],
        '松江': [['松江', '松江区'], 121.23, 31.03],
        '虹桥': [['虹桥'], 121.38, 31.20],
        '张江': [['张江'], 121.58, 31.20],
        '临港': [['临港'], 121.88, 30.90]
    }

    file_paths = [
        r"data\2122_t_f.csv",
        r"data\2223_t_f.csv",
        r"data\2324_t_f.csv",
        r"data\2425_t_f.csv"
    ]

    combined_df = pd.concat([pd.read_csv(fp) for fp in file_paths], ignore_index=True)

    place_freq = {}
    for place_name, (variants, lon, lat) in shanghai_places.items():
        place_freq[place_name] = {
            'frequency': combined_df[combined_df['word'].isin(variants)]['frequency'].sum(),
            'longitude': lon,
            'latitude': lat
        }

    place_freq_df = pd.DataFrame.from_dict(place_freq, orient='index')
    place_freq_df = place_freq_df[place_freq_df['frequency'] > 0]
    place_freq_df.reset_index(inplace=True)
    place_freq_df.rename(columns={'index': 'place'}, inplace=True)

    fig = px.scatter_geo(place_freq_df,
                         lat='latitude',
                         lon='longitude',
                         size='frequency',
                         hover_name='place',
                         hover_data={'frequency': True, 'latitude': False, 'longitude': False},
                         scope='asia',
                         title='Shanghai Word Frequency Map (2021-2025)')

    fig.update_geos(
        resolution=50,
        center=dict(lat=31.23, lon=121.47),
        projection_scale=20,
        showcountries=True,
        showsubunits=True,
        landcolor='lightgray',
        oceancolor='lightblue'
    )

    fig.update_layout(
        width=800,
        height=800
    )

    return fig


def create_parallel_categories():
    """Create parallel categories visualization of community evolution"""
    years = ['2122', '2223', '2324', '2425']
    dataframes = {}
    partitions = {}

    for year in years:
        file_path = f"data/{year}_matrix_article.csv"
        dataframes[year] = pd.read_csv(file_path)
        G = generate_undirected_graph(dataframes[year])
        partitions[year] = detect_communities(G)

        # Standardize community IDs
        unique_ids = sorted(set(partitions[year].values()))
        id_mapping = {old: new for new, old in enumerate(unique_ids)}
        partitions[year] = {k: id_mapping[v] for k, v in partitions[year].items()}

    # Create node data
    entity_data = {}
    for year in years:
        data = dataframes[year]
        partition = partitions[year]

        for entity in data['Entity1'].unique():
            total_cooccurrence = data[(data['Entity1'] == entity) | (data['Entity2'] == entity)]['CoOccurrence'].sum()
            size = min(4 + total_cooccurrence * 0.4, 10)
            community_id = partition.get(entity, -1)

            if entity not in entity_data:
                entity_data[entity] = {
                    'sizes': {year: size},
                    'communities': {year: community_id}
                }
            else:
                entity_data[entity]['sizes'][year] = size
                entity_data[entity]['communities'][year] = community_id

    # Convert to DataFrame
    df = pd.DataFrame([
        {'entity': entity, **{f'community_{year}': data['communities'].get(year, -1) for year in years}}
        for entity, data in entity_data.items()
    ])

    # Create figure
    community_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    community_names = {0: "技术", 1: "资本", 2: "产业"}

    fig = go.Figure(go.Parcats(
        dimensions=[
            {'label': f'20{year[:2]}-20{year[2:]}',
             'values': df[f'community_{year}'],
             'categoryarray': [0, 1, 2]
             } for year in years
        ],
        line={
            'color': [community_colors[cid] if cid in [0, 1, 2] else '#CCCCCC'
                      for cid in df[f'community_{years[0]}']],
            'shape': 'hspline'
        },
        labelfont={'size': 14, 'family': 'SimHei'},
        arrangement='freeform'
    ))

    # Add legend
    for i, color in enumerate(community_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=community_names[i],
            showlegend=True
        ))

    fig.update_layout(
        title='G60 Communities Evolution (2021-2025)',
        width=1200,
        height=800,
        legend=dict(title='社区分类', orientation='h', y=-0.15)
    )

    return fig



def create_rd_trend_plot():
    """创建与研发共现实体的趋势图，匹配原matplotlib版本的样式"""
    file_paths = {
        "2122": r"data\2122_matrix_article.csv",
        "2223": r"data\2223_matrix_article.csv",
        "2324": r"data\2324_matrix_article.csv",
        "2425": r"data\2425_matrix_article.csv"
    }

    # 加载所有年份数据
    all_data = pd.concat([pd.read_csv(path).assign(Year=year) for year, path in file_paths.items()])

    # 2. 处理与"研发"共现的数据
    # 确保"研发"始终在Entity1列
    investment_data = all_data[(all_data['Entity1'] == '研发') | (all_data['Entity2'] == '研发')].copy()
    mask = investment_data['Entity2'] == '研发'
    investment_data.loc[mask, ['Entity1', 'Entity2']] = investment_data.loc[mask, ['Entity2', 'Entity1']].values

    # 3. 获取与"研发"共现最多的10个实体
    top_entities = investment_data.groupby('Entity2')['CoOccurrence'].sum().nlargest(10).index.tolist()

    # 4. 构建共现网络并检测社区
    filtered_data = all_data[
        (all_data['Entity1'].isin(top_entities + ['研发'])) &
        (all_data['Entity2'].isin(top_entities + ['研发']))
        ]
    G = generate_undirected_graph(filtered_data)
    partition = detect_communities(G)

    # 5. 准备绘图数据
    plot_data = []
    for year, year_data in investment_data.groupby('Year'):
        for entity in top_entities:
            co_occurrence = year_data[year_data['Entity2'] == entity]['CoOccurrence'].sum()
            community_id = partition.get(entity, -1)
            plot_data.append({
                'Year': year,
                'Entity': entity,
                'CoOccurrence': co_occurrence,
                'Community': f'社区 {community_id + 1}'  # 使用中文
            })

    plot_df = pd.DataFrame(plot_data)

    # 6. 创建可视化图表
    fig = go.Figure()

    # 使用与matplotlib相似的配色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # 为每个实体添加折线
    for i, entity in enumerate(plot_df['Entity'].unique()):
        entity_data = plot_df[plot_df['Entity'] == entity]
        community = entity_data['Community'].iloc[0]
        color_idx = int(community.split()[-1]) - 1  # 获取社区索引用于配色

        fig.add_trace(go.Scatter(
            x=entity_data['Year'],
            y=entity_data['CoOccurrence'],
            mode='lines+markers',
            name=entity,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            marker=dict(symbol='circle', size=8),
            legendgroup=community,
            hovertemplate=f"<b>{entity}</b><br>年份: %{{x}}<br>共现次数: %{{y}}<extra></extra>"
        ))

    # 7. 设置图表布局
    fig.update_layout(
        title=dict(
            text='Top Entities Co-occurring with "研发" (2021-2025)',
            font=dict(size=18, family='SimHei'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='年份',
            tickmode='array',
            tickvals=['2122', '2223', '2324', '2425'],
            ticktext=['2021-2022', '2022-2023', '2023-2024', '2024-2025'],
            title_font=dict(family='SimHei')
        ),
        yaxis=dict(
            title='共现次数',
            title_font=dict(family='SimHei')
        ),
        legend=dict(
            title=dict(text='社区/实体', font=dict(family='SimHei')),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        width=1000,
        height=600,
        plot_bgcolor='white',
        hovermode='x unified',
        font=dict(family='SimHei')  # 全局中文字体设置
    )

    # 8. 添加社区分组到图例
    communities = sorted(plot_df['Community'].unique())
    for i, community in enumerate(communities):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=colors[i % len(colors)]),
            name=community,
            legendgroup=community,
            showlegend=True
        ))

    return fig


def create_radar_chart():
    """Create radar chart matching the matplotlib version"""
    # Read data
    df = read_entity_dict()

    # Separate data by category
    industry_data = df[df['category'] == '产业']
    technology_data = df[df['category'] == '技术']
    capital_data = df[df['category'] == '资本']

    # Extract entities and frequencies
    industries = industry_data['entity'].tolist()
    frequencies_industry = industry_data['frequency'].tolist()

    technologies = technology_data['entity'].tolist()
    frequencies_technology = technology_data['frequency'].tolist()

    capitals = capital_data['entity'].tolist()
    frequencies_capital = capital_data['frequency'].tolist()

    # Combine all categories for the radar axes
    categories = industries + technologies + capitals
    N = len(categories)

    # Create values for each category (0 for other categories' positions)
    values_industry = frequencies_industry + [0] * (len(technologies) + len(capitals))
    values_technology = [0] * len(industries) + frequencies_technology + [0] * len(capitals)
    values_capital = [0] * (len(industries) + len(technologies)) + frequencies_capital

    # Define the same non-linear scaling function
    def scale_values(x):
        if x <= 2000:
            return x
        else:
            return 2000 + (x - 2000) / 5

    # Apply scaling
    scaled_industry = [scale_values(v) for v in values_industry]
    scaled_technology = [scale_values(v) for v in values_technology]
    scaled_capital = [scale_values(v) for v in values_capital]

    # Create Plotly figure
    fig = go.Figure()

    # Add traces with the same colors and styling
    fig.add_trace(go.Scatterpolar(
        r=scaled_industry + [scaled_industry[0]],  # Close the loop
        theta=categories + [categories[0]],  # Close the loop
        fill='toself',
        name='Industry',
        line=dict(color='#ff9999', width=2),
        fillcolor='rgba(255, 153, 153, 0.25)',
        hoverinfo='r+theta+name'
    ))

    fig.add_trace(go.Scatterpolar(
        r=scaled_technology + [scaled_technology[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Technology',
        line=dict(color='#66b3ff', width=2),
        fillcolor='rgba(102, 179, 255, 0.25)',
        hoverinfo='r+theta+name'
    ))

    fig.add_trace(go.Scatterpolar(
        r=scaled_capital + [scaled_capital[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Capital',
        line=dict(color='#99ff99', width=2),
        fillcolor='rgba(153, 255, 153, 0.25)',
        hoverinfo='r+theta+name'
    ))

    # Configure layout to match matplotlib version
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickvals=[0, 500, 1000, 1500, 2000, 3000],
                ticktext=['0', '500', '1000', '1500', '2000', '3000'],
                range=[0, max(max(scaled_industry), max(scaled_technology), max(scaled_capital))],
                gridcolor='#d3d3d3',
                gridwidth=0.5,
                showline=False,
                ticksuffix='',
                angle=0
            ),
            angularaxis=dict(
                rotation=90,
                direction='clockwise',
                gridcolor='#d3d3d3',
                gridwidth=0.5,
                showline=False,
                tickfont=dict(size=10)
            ),
            bgcolor='#f9f9f9'
        ),
        title=dict(
            text='G60 Word Frequency Radar',
            font=dict(size=14)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        width=800,
        height=800,
        paper_bgcolor='white',
        font=dict(family="SimHei")  # For Chinese character support
    )

    return fig


def create_stacked_bar():
    """创建按类别分组的实体频率堆叠柱状图，匹配原Python文件中的样式"""

    # 1. 数据准备
    df = read_entity_dict()

    # 2. 颜色生成
    # 定义基础颜色（与原文件一致）
    base_colors = {
        '产业': '#FFA07A',  # 浅珊瑚色
        '技术': '#ADD8E6',  # 浅蓝色
        '资本': '#98FB98'  # 浅绿色
    }

    # 生成渐变颜色
    def generate_gradient_colors(base_color, num_colors):
        """生成渐变色，与原Python文件中的算法一致"""
        base_rgb = hex_to_rgb(base_color)
        gradient_colors = []
        for i in range(num_colors):
            factor = i / (num_colors - 1) if num_colors > 1 else 0
            new_color = (
                min(255, int(base_rgb[0] * (1 + factor * 0.2))),
                min(255, int(base_rgb[1] * (1 + factor * 0.2))),
                min(255, int(base_rgb[2] * (1 + factor * 0.2)))
            )
            gradient_colors.append(f'rgb{new_color}')
        return gradient_colors

    # 为每个类别生成渐变色
    color_map = {}
    for category in df['category'].unique():
        category_entities = df[df['category'] == category]
        num_entities = len(category_entities)
        colors = generate_gradient_colors(base_colors[category], num_entities)

        # 按频率排序实体，使颜色渐变与高度对应
        sorted_entities = category_entities.sort_values('frequency', ascending=False)['entity']
        for entity, color in zip(sorted_entities, colors):
            color_map[entity] = color

    # 3. 创建图表
    fig = px.bar(
        df,
        x='category',
        y='frequency',
        color='entity',
        barmode='stack',
        color_discrete_map=color_map,
        labels={'category': '类别', 'frequency': '频率'},
        title='<b>堆叠柱状图 - 按类别分组的实体频率</b>'
    )

    # 4. 布局调整（匹配原Python文件中的样式）
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_x=0.5,  # 标题居中
        width=1200,
        height=800,
        font=dict(family="SimHei"),  # 支持中文
        legend_title_text='<b>实体</b>',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=12),
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal'  # 保持图例项顺序一致
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="SimHei"
        ),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )

    # 5. 添加中文支持
    fig.update_layout(
        title_font_family="SimHei",
        legend_title_font_family="SimHei",
        font=dict(family="SimHei")
    )

    return fig
def create_combined_html():
    """Combine all visualizations into a single HTML file"""
    # Previous visualizations
    sankey_fig = create_sankey_diagram()
    bar_fig = create_parallel_bar_chart()
    line_fig = create_line_chart()
    cooccurrence_fig = create_cooccurrence_network()
    force_directed_fig = create_force_directed_network()
    heatmap_fig = create_heatmap()
    changjiang_map_fig = create_changjiang_map()
    shanghai_map_fig = create_shanghai_map()
    parallel_cat_fig = create_parallel_categories()
    rd_trend_fig = create_rd_trend_plot()
    stacked_bar_fig = create_stacked_bar()
    radar_fig = create_radar_chart()

    # Create a combined HTML with all figures
    with open('combined_visualizations.html', 'w') as f:
        f.write('<html><head><title>Combined Visualizations</title></head><body>')
        f.write('<h1 style="text-align:center">Combined Visualizations</h1>')

        f.write('<h2 style="text-align:center">Co-occurrence of Entities Over Four Years (Sankey Diagram)</h2>')
        f.write(sankey_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Community Detection in Relationship Networks (Parallel Bar Chart)</h2>')
        f.write(bar_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">G60 Word Frequency Changes (Line Chart)</h2>')
        f.write(line_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Co-occurrence Network Analysis</h2>')
        f.write(cooccurrence_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Force-Directed Network of Word Evolution</h2>')
        f.write(force_directed_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Four-Year Co-occurrence Matrix Heatmap</h2>')
        f.write(heatmap_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # New visualizations
        f.write('<h2 style="text-align:center">Changjiang Delta Word Frequency Map</h2>')
        f.write(changjiang_map_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Shanghai Word Frequency Map</h2>')
        f.write(shanghai_map_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Community Evolution Parallel Categories</h2>')
        f.write(parallel_cat_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        # New visualizations sections
        f.write('<h2 style="text-align:center">R&D Co-occurrence Trend</h2>')
        f.write(rd_trend_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Entity Frequency by Category (Stacked Bar)</h2>')
        f.write(stacked_bar_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('<h2 style="text-align:center">Word Frequency Radar Chart</h2>')
        f.write(radar_fig.to_html(full_html=False, include_plotlyjs='cdn'))

        f.write('</body></html>')

    print("Combined HTML file created: combined_visualizations.html")


if __name__ == "__main__":
     create_combined_html()
