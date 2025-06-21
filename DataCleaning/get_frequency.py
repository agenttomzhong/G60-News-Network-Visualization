import pandas as pd
from collections import Counter

def get_frequency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Count word frequency, filter by length
    words = []
    tokens = df['tokens'].dropna().astype(str).tolist()
    for text in tokens:
        for word in text.split():
            if len(word) == 1:
                continue
            elif len(word) >= 9:
                continue
            else: 
                words.append(word)
    frequency = Counter(words)
    return pd.DataFrame(frequency.items(), columns=[column, 'frequency']).sort_values(by='frequency', ascending=False)

def load_csv(file_path: str) -> pd.DataFrame:
    # Load CSV file
    return pd.read_csv(file_path, encoding='utf-8-sig')
def save_csv(df: pd.DataFrame, file_path: str) -> None:
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # Main execution: load, process, and save frequency
    input_csv =r"D:\Documents\Python\Proj25.06.04\WorkData\tokenlised\2324_t.csv"
    input_name = input_csv.split('\\')[-1]
    output_csv = f'DataCleaning\{input_name[:-4]}_f.csv'
    df = load_csv(input_csv)
    if 'tokens' not in df.columns:
        raise ValueError("Column 'tokens' must exist in CSV")
    frequency_df = get_frequency(df, 'word')
    save_csv(frequency_df, output_csv)
    print(f"Word frequency counted and saved to {output_csv}")
