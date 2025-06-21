from graphviz import Digraph
import html

# 定义主题色（按子图分类）
COLORS = {
    'preprocessing': {
        'header': '#4A86E8',  # 深蓝
        'item': '#D9E7FF',    # 浅蓝
        'text': 'white'
    },
    'analysis': {
        'header': '#E67E22',  # 橙色
        'item': '#FFF5E4',    # 浅橙
        'text': 'black'
    },
    'network': {
        'header': '#8E44AD',  # 紫色
        'item': '#F5E6FA',    # 浅紫
        'text': 'white'
    }
}

def create_html_label(main_title, sub_items, theme='preprocessing'):
    """生成带主题色和圆角的HTML表格"""
    # 转义特殊字符
    main_title = main_title.replace('&', '&amp;')
    sub_items = [html.escape(item) for item in sub_items]
    
    # 获取主题颜色
    theme_colors = COLORS.get(theme, COLORS['preprocessing'])
    header_bg = theme_colors['header']
    item_bg = theme_colors['item']
    text_color = theme_colors['text']
    
    # 构建表格行
    table_rows = '\n'.join([
        f'<TR><TD ALIGN="LEFT" BGCOLOR="{item_bg}" STYLE="padding: 8px;">{item}</TD></TR>' 
        for item in sub_items
    ])
    
    return f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" 
        STYLE="border-radius: 12px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <TR>
        <TD ALIGN="CENTER" BGCOLOR="{header_bg}" 
            STYLE="padding: 12px; color: {text_color}; font-size: 18px;">
            <B>{main_title}</B>
        </TD>
    </TR>
    {table_rows}
</TABLE>
>'''

def create_workflow_diagram():
    dot = Digraph(
        comment='News to Network Workflow',
        format='png',
        engine='dot',
        graph_attr={
            'rankdir': 'LR',
            'fontsize': '20',
            'fontname': 'Microsoft YaHei',
            'margin': '0.5',
            'ranksep': '3.5',
            'nodesep': '1.5',
            'splines': 'spline',
            'ratio': 'auto',
            'bgcolor': '#F0F8FF',
            'dpi': '400'
        },
        node_attr={
            'shape': 'rounded',
            'style': 'filled',
            'fillcolor': '#F0F8FF',
            'fontname': 'Microsoft YaHei',
            'fontsize': '18',
            'color': '#FFFFFF',      # 移除外层节点边框
            'height': '0.8'
        },
        edge_attr={
            'fontname': 'Microsoft YaHei',
            'fontsize': '16',
            'arrowsize': '0.7',
            'color': '#333333',
            'penwidth': '1.2'
        }
    )

    # 主步骤节点（使用主题色）
    dot.node('A', label=create_html_label('A. Data Collection & Cleaning', [
        'A1. Source Selection', 
        'A2. Keyword Retrieval',
        'A3. Noise Filtering',
        'A4. Manual Validation'
    ], theme='preprocessing'))
    
    dot.node('B', label=create_html_label('B. Data Structuring', [
        'B1. Text Parsing', 
        'B2. Metadata Extraction',
        'B3. Data Cleaning',
        'B4. CSV Export'
    ], theme='preprocessing'))
    
    dot.node('C', label=create_html_label('C. Core Entity Dictionary', [
        'C1. Word Segmentation', 
        'C2. Frequency Analysis',
        'C3. LLM Screening',
        'C4. Manual Validation'
    ], theme='analysis'))
    
    dot.node('D', label=create_html_label('D. Entity Relation Extraction', [
        'D1. Sentence Segmentation', 
        'D2. Entity Matching',
        'D3. Co-occurrence Pairing'
    ], theme='analysis'))
    
    dot.node('E', label=create_html_label('E. Co-occurrence Network', [
        'E1. Matrix Construction', 
        'E2. Threshold Filtering',
        'E3. Network Building'
    ], theme='network'))
    
    dot.node('F', label=create_html_label('F. Network Analysis & Visualization', [
        'F1. Centrality Calculation', 
        'F2. Community Detection',
        'F3. Multi-view Visualization'
    ], theme='network'))

    # 连接主节点
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

    # 子图集群
    with dot.subgraph(name='cluster_preprocessing') as pre:
        pre.attr(
            label='\nPreprocessing Phase',
            style='filled,rounded',
            color='#A9CCE3',
            fillcolor='#EBF5FB',
            fontsize='40',
            fontname='Microsoft YaHei Bold',
            margin='80,30',
            penwidth='6'
        )
        pre.node('A')
        pre.node('B')

    with dot.subgraph(name='cluster_analysis') as analysis:
        analysis.attr(
            label='\nEntity Analysis',
            style='filled,rounded',
            color='#F9E79F',
            fillcolor='#FEF9E7',
            fontsize='40',
            fontname='Microsoft YaHei Bold',
            margin='80,30',
            penwidth='6'
        )
        analysis.node('C')
        analysis.node('D')

    with dot.subgraph(name='cluster_network') as network:
        network.attr(
            label='\nNetwork Science',
            style='filled,rounded',
            color='#D7BDE2',
            fillcolor='#F8F6F6',
            fontsize='40',
            fontname='Microsoft YaHei Bold',
            margin='80,30',
            penwidth='6'
        )
        network.node('E')
        network.node('F')

    return dot

# 生成并渲染图表
if __name__ == '__main__':
    workflow = create_workflow_diagram()
    workflow.render('news_network_workflow_optimized', view=True, cleanup=True)