import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    total_restaurant_info = []
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            tmp_df = pd.read_csv(file_path + file)
            restaurant_info = {tmp_df.iloc[i, 0]: tmp_df.iloc[i, 1] for i in range(len(tmp_df))}
            total_restaurant_info.append(restaurant_info)
    return total_restaurant_info
            

def preprocess_data(df, tokenizer):
    return tokenizer(df['text'], truncation=True, padding='max_length')