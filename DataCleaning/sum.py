import pandas as pd
import re
import glob
import os

def is_not_digit(word):
    # Return True if not a pure digit
    return not re.fullmatch(r'\d+', str(word))

def merge_and_sum_csvs(folder_path):
    # Get all csv files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    file_prefixes = []
    for file in csv_files:
        prefix = os.path.basename(file)[:4]
        file_prefixes.append(prefix)
        df = pd.read_csv(file)
        # Add a new column with prefix, value is frequency
        df[prefix] = df['frequency']
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    # Filter out rows where 'word' is pure digit
    all_df = all_df[all_df['word'].apply(is_not_digit)]
    # Prepare columns for grouping and summing
    group_cols = ['word', 'frequency'] + [col for col in set(file_prefixes) if col in all_df.columns]
    result = all_df.groupby('word', as_index=False)[group_cols[1:]].sum()
    return result

if __name__ == "__main__":
    folder = os.path.dirname(__file__)
    merged_df = merge_and_sum_csvs(folder)
    merged_df = merged_df.sort_values(by='frequency', ascending=False)
    # Save merged and summed result
    merged_df.to_csv(os.path.join(folder, "merged_sum.csv"), index=False, encoding='utf-8-sig')
    print("Merge and sum completed, result saved to merged_sum.csv")

