from torch.utils.data import Dataset

import os
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def load_data(file_path: str) -> list:
    total_restaurant_info = []
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            tmp_df = pd.read_csv(file_path + file)
            restaurant_info = ', '.join(f'{row.iloc[0]}: "{row.iloc[1]}"' for _, row in tmp_df.iterrows())
            total_restaurant_info.append(restaurant_info)
    return total_restaurant_info
            

def preprocess_data(data: list[str], tokenizer) -> dict:
    return CustomDataset(tokenizer(data, truncation=True, padding='max_length', max_length=1024, return_tensors='pt'))