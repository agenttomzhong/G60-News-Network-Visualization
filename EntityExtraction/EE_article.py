import re
import pandas as pd
from collections import Counter
from logging import getLogger
import jieba

INPUT_FILE = r"D:\Documents\Python\Proj25.06.04\WorkData\tokenlised\2425_t.csv"
ENTITY_DICT = pd.read_csv("EntityExtraction/EntityDict/result/entity_dict.csv", dtype=str, encoding='utf-8-sig')
logger = getLogger('EntityExtraction')


def load_entities(category=None):
    # Filter entities by category column
    return set(ENTITY_DICT[ENTITY_DICT['category'] == category]['entity'].tolist())

industries = load_entities("产业")
technologies = load_entities("技术")
capitals = load_entities("资本")


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


def load_articles_tokenized(csv_path):
    # Load articles and tokenize
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df['cleaned_text'].dropna().tolist()
    jieba.load_userdict(r'WorkData\entity_dict.txt')
    tokenized_articles = []
    for text in texts:
        # Sentence splitting
        sentences = re.split(r"([。！？\.!?])", text)
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        # Tokenize and remove stopwords
        words = []
        for sent in sentences:
            if sent.strip():
                segs = jieba.lcut(sent)
                segs = remove_stopwords(" ".join(segs)).split()
                words.extend(segs)
        tokenized_articles.append(words)
    return tokenized_articles  # List[List[str]], each article is a list of words


def extract_entity_pairs_by_article(tokenized_articles):
    # Extract entity pairs from each article
    entity_pairs = []
    for words in tokenized_articles:
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

def build_co_matrix(entity_pairs):
    # Build co-occurrence matrix from entity pairs
    co_matrix = Counter()
    co_matrix.update(entity_pairs)
    rows = []
    for (ent1, ent2), count in co_matrix.items():
        rows.append({"Entity1": ent1, "Entity2": ent2, "CoOccurrence": count})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Load news data (one article per item)
    tokenized_articles = load_articles_tokenized(INPUT_FILE)
    # Extract entity pairs
    entity_pairs = extract_entity_pairs_by_article(tokenized_articles)
    # Build co-occurrence matrix
    co_matrix_df = build_co_matrix(entity_pairs)
    # Output to CSV
    co_matrix_df.to_csv("EntityExtraction\\co_occurrence_matrix_article.csv", index=False, encoding='utf-8-sig')
    print("Co-occurrence matrix saved as EntityExtraction\\co_occurrence_matrix_article.csv")
