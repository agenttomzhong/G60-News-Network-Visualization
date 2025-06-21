import pandas as pd
import re


def preprocess_text_file(file_path):
    """
    从文本文件加载原始新闻内容，分块提取字段，生成DataFrame。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    chunks = re.split(r'\n(?=\d+\.)', raw)
    chunks = [c.strip() for c in chunks if c.strip()]
    records = []
    for chunk in chunks:
        chunk = re.sub(r'^\d+\.\s*', '', chunk)
        chunk = re.sub(r'^文字快照.*$', '', chunk, flags=re.MULTILINE)
        chunk = re.sub(
            r"所有文章或连结中所列载[\s\S]*?责任。",
            "",
            chunk
        )
        lines = chunk.splitlines()
        header_line = ''
        for line in lines:
            if '(' in line and ')' in line and re.search(r"\d{4}-\d{2}-\d{2}", line):
                header_line = line.strip()
                break
        secondary_source = ''
        date = ''
        title = ''
        if header_line:
            if ' - ' in header_line:
                secondary_source = header_line.split(' - ')[0].strip()
            m_date = re.search(r"(\d{4}-\d{2}-\d{2})", header_line)
            if m_date:
                date = m_date.group(1)
            m_title = re.search(r"\(([^)]+)\)", header_line)
            if m_title:
                title = m_title.group(1).strip()
            chunk = chunk.replace(header_line, '').strip()
        chunk = re.sub(r"文字快照：http://\S+", "", chunk)
        chunk = re.sub(r"资源均由有关媒体.*?(?:$|文章编号)", "", chunk, flags=re.DOTALL)
        chunk = re.sub(r"文章编号:\s*\[\d+\]", "", chunk)
        chunk = re.sub(r"^[-–—]{2,}\s*$", "", chunk, flags=re.MULTILINE)
        m_src = re.search(r"来源[:：]\s*([^\n]+)", chunk)
        source = m_src.group(1).strip() if m_src else secondary_source
        chunk = re.sub(r"来源[:：]\s*[^\n]+", "", chunk)
        if date:
            chunk = chunk.replace(date, '')
        if title:
            chunk = chunk.replace(title, '', 1)
            chunk = re.sub(r"[()]", "", chunk)
        content = re.sub(r"\s+", " ", chunk).strip()
        if content == '没有文字档。':
            continue
        records.append({
            '日期': date,
            '来源': source,
            '标题': title,
            '内容': content
        })
    df = pd.DataFrame(records, columns=['日期', '来源', '标题', '内容'])
    return df


def generate_id_and_sort(df):
    """
    生成编号，排序并整理字段顺序。
    """
    df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['年'] = df['日期'].dt.year.astype(str).str[-2:]
    df['半年'] = df['日期'].dt.month.apply(lambda m: '01' if m <= 6 else '02')
    df['序号'] = (
        df
        .groupby(['年', '半年'], sort=False)
        .cumcount()
        .add(1)
        .astype(str)
        .str.zfill(2)
    )
    df['编号'] = df['年'] + df['半年'] + df['序号']
    df.drop(columns=['年', '半年', '序号'], inplace=True)
    df.sort_values('编号', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
    df = df[['编号', '日期', '来源', '标题', '内容']]
    return df


def clean_final_columns(df):
    """
    清洗编号、来源、内容列。
    """
    # 编号列格式化为 'YYYY/MM
    df['编号'] = "'" + (
        df['编号'].str.extract(r'(\d{6})')[0].str[:4] + '/' +
        df['编号'].str.extract(r'(\d{6})')[0].str[4:6]
    )
    # 来源列去除"APP"
    df['来源'] = df['来源'].str.replace('APP', '', regex=False)
    # 内容列进一步清洗
    def clean_content(text):
        if pd.isna(text):
            return text
        text = re.sub(r'^本报.*?电\（记者.*?\）', '', text)
        text = re.sub(r'^当前位置： 首页 > 新闻中心 > .*? > 正文', '', text)
        text = text.replace("\n", "").replace(" * ", "")
        return text.strip()
    df['内容'] = df['内容'].apply(clean_content)
    return df


def main():
    # 步骤1：文本预处理
    input_txt = r'DataCleaning\\news.txt'
    df = preprocess_text_file(input_txt)
    # 步骤2：生成编号并排序
    df = generate_id_and_sort(df)
    # 步骤3：清洗编号、来源、内容
    df = clean_final_columns(df)
    # 步骤4：输出
    output_csv = r'DataCleaning\\output.csv'
    df.to_csv(output_csv, index=False, encoding='utf_8_sig')
    print(f'已生成文件：{output_csv}')


if __name__ == "__main__":
    main()