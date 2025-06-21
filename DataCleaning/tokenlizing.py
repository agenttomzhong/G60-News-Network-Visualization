import re
import pandas as pd
import jieba
from logging import getLogger

logger = getLogger('DataCleaning.tokenlizing')

HTML_TAG_RE = re.compile(r'<[^<]+?>')
PUNCTUATION_RE = re.compile(r'[^\w\s]')
WHITESPACE_RE = re.compile(r'\s+')

def _remove_html_tags(text: str) -> str:
    # Remove HTML tags
    return HTML_TAG_RE.sub('', text)
def _normalize_whitespace(text: str) -> str:
    # Normalize whitespace
    return WHITESPACE_RE.sub(' ', text).strip()

def clean_text(text: str) -> tuple:
    # Clean text: remove HTML, keep basic punctuation, remove extra whitespace
    text = _remove_html_tags(text)

    # Keep basic punctuation
    cleaned_text = re.sub(r'[^\w\s\u3000-\u303F\uFF00-\uFFEF。，！？、]+', '', text)
    cleaned_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', cleaned_text).strip()

    # Remove all punctuation for full clean
    text = PUNCTUATION_RE.sub('', text)
    text = _normalize_whitespace(text)

    return cleaned_text, text


def load_stopwords() -> set:
    stopwords = set()
    stopwords_file = (r'D:\Documents\Python\Proj25.06.04\DataCleaning\stopwords.txt')
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords.update(line.strip().lower() for line in f if line.strip())
        logger.debug(f"Loaded {len(stopwords)} stopwords")
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found. Continuing without stopwords.")
    return stopwords
def remove_stopwords(text: str) -> str:
    stopwords = load_stopwords()
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    logger.debug(f"Removed {len(words) - len(filtered_words)} stopwords from text.")
    return " ".join(filtered_words)

def process_text(df: pd.DataFrame) -> pd.DataFrame:
    """处理DataFrame中的文本数据"""
    if df.shape[1] == 1 and df.columns[0] != 'text':
        df.columns = ['text']
    elif 'text' not in df.columns:
        raise ValueError("Input DataFrame must contain a column named 'text'.")

    df = df.drop_duplicates(subset=['text'])
    logger.info(f"Deduplicated DataFrame: {len(df)} rows.")

    processed_data = []
    for idx, row in df.iterrows():
        text = row['text'].strip() if isinstance(row['text'], str) else ""
        if not text:
            logger.warning(f"Empty text found at row {idx}. Skipping...")
            continue

        try:
            cleaned_text, fully_cleaned = clean_text(text)
            jieba.load_userdict(r'DataCleaning\user_dict.txt')
            tokens = jieba.lcut(fully_cleaned)
            filtered_tokens = remove_stopwords(" ".join(tokens)).split()

            processed_data.append({
                'cleaned_text': cleaned_text,
                'tokens': " ".join(filtered_tokens)
            })
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")

    processed_df = df.reset_index(drop=True).copy()
    processed_cols = pd.DataFrame(processed_data)
    processed_df = pd.concat([processed_df, processed_cols], axis=1)
    return processed_df

if __name__ == "__main__":
    # 读取CSV文件
    input_csv =r'D:\Documents\Python\Proj25.06.04\WorkData\dataset\2122.csv' 
    output_csv = './DataCleaning/output.csv'
    df = pd.read_csv(input_csv, encoding='utf-8')
    # 只保留text列
    if 'text' not in df.columns:
        raise ValueError("CSV文件中必须包含'text'列")
    df_1 = df[['text']]
    # 处理文本
    processed_df = process_text(df_1)
    processed_df = pd.concat([df, processed_df], axis=1)
    processed_df = processed_df.drop(columns=['text'])
    # 保存为utf-8-sig编码的CSV文件
    processed_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"处理完成，已保存到 {output_csv}")

