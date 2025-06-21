import re
import pandas as pd
from collections import defaultdict, Counter
from logging import getLogger
import jieba

INPUT_FILE = r"D:\Documents\Python\Proj25.06.04\WorkData\tokenlised\2425_t.csv"
ENTITY_DICT = pd.read_csv("EntityExtraction/EntityDict/result/entity_dict.csv", dtype=str, encoding='utf-8-sig')
logger = getLogger('EntityExtraction')


# Load entity dictionary by category
def load_entities(category=None):
    # Filter entities by category column
    return set(ENTITY_DICT[ENTITY_DICT['category'] == category]['entity'].tolist())

industries = load_entities("产业")
technologies = load_entities("技术")
capitals = load_entities("资本")

# Load and tokenize news data
def load_news_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df['cleaned_text'].dropna().tolist()
    all_sentences = []
    for text in texts:
        # Split text into sentences
        sentences = split_sentences(text)
        # Tokenize sentences
        jieba.load_userdict(r'WorkData\entity_dict.txt')
        tokenized_sentences = [jieba.lcut(sent) for sent in sentences if sent.strip()]
        tokenized_sentences = [remove_stopwords(" ".join(words)).split() for words in tokenized_sentences]
        all_sentences.extend(tokenized_sentences)
    return all_sentences  # Return list of tokenized sentences

# Split text into sentences using punctuation
def split_sentences(text):
    sentences = re.split(r"([。！？\.!?])", text)
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    return sentences

def load_stopwords() -> set:
    stopwords = set()
    stopwords_file = (r'DataCleaning\stopwords.txt')
    try:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords.update(line.strip().lower() for line in f if line.strip())
        logger.debug(f"Loaded {len(stopwords)} stopwords")
    except FileNotFoundError:
        logger.warning(f"Stopwords file not found. Continuing without stopwords.")
    return stopwords

def remove_stopwords(text: str) -> str:
    # Remove stopwords from text
    stopwords = load_stopwords()
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    logger.debug(f"Removed {len(words) - len(filtered_words)} stopwords from text.")
    return " ".join(filtered_words)

# Extract entity pairs from tokenized sentences
def extract_entity_pairs(tokenized_sentences):
    entity_pairs = []
    for words in tokenized_sentences:
        found_entities = []
        for word in words:
            if word in industries:
                found_entities.append((word, "industry"))
            elif word in technologies:
                found_entities.append((word, "technology"))
            elif word in capitals:
                found_entities.append((word, "capital"))
        for i in range(len(found_entities)):
            for j in range(i+1, len(found_entities)):
                ent1, cat1 = found_entities[i]
                ent2, cat2 = found_entities[j]
                if cat1 != cat2:
                    if ent1 < ent2:
                        entity_pairs.append((ent1, ent2))
                    else:
                        entity_pairs.append((ent2, ent1))
    return entity_pairs

# Build co-occurrence matrix from entity pairs
def build_co_matrix(tokenized_sentences):
    co_matrix = Counter()
    pairs = extract_entity_pairs(tokenized_sentences)
    co_matrix.update(pairs)
    rows = []
    for (ent1, ent2), count in co_matrix.items():
        rows.append({"Entity1": ent1, "Entity2": ent2, "CoOccurrence": count})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Load news data
    tokenized_sentences = load_news_data(INPUT_FILE)
    # Build co-occurrence matrix
    co_matrix_df = build_co_matrix(tokenized_sentences)
    # Output to CSV
    co_matrix_df.to_csv("EntityExtraction\co_occurrence_matrix.csv", index=False, encoding='utf-8-sig')
    print("Co-occurrence matrix saved as co_occurrence_matrix.csv")
