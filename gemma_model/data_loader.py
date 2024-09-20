import os
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.inputs = tokenizer(
            data_list,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.inputs['input_ids'][idx],
        }


def generate_prompt(question, answer):
    prompt = f"""<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn><eos>"""
    return prompt


def load_data(file_path):
    data_list = []
    
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(file_path, file), header=None)
            data = {row[0].strip('"'): row[1].strip('"') for _, row in df.iterrows()}
            restaurant_name = data.get('음식점명', '알 수 없는 음식점')
            menu = data.get('취급메뉴', '정보 없음')
            address = data.get('주소', '정보 없음')
            overview = data.get('개요', '정보 없음')
            review_keywords = data.get('리뷰키워드', '정보 없음')
            parking = data.get('주차시설 유무', '정보 없음')
            hours = data.get('영업시간', '정보 없음')
            holiday = data.get('휴무일', '정보 없음')
            
            # 1. 특정 지역과 메뉴에 따른 맛집 추천 질문
            location = ' '.join(address.split()[:3])  # 주소의 앞 세 단어 사용
            question = f"{location}에서 {menu}를 먹고 싶은데, 추천해줄 만한 곳 있어?"
            answer = f"{restaurant_name}을 추천합니다! 주소는 {address}이고, 영업시간은 {hours}입니다. 주 메뉴는 {menu}이며, {overview} 리뷰 키워드는 {review_keywords}입니다."
            prompt = generate_prompt(question, answer)
            data_list.append(prompt)
            
            # 2. 음식점의 영업시간 질문
            question = f"{restaurant_name}의 영업시간이 어떻게 되나요?"
            answer = f"{restaurant_name}의 영업시간은 {hours}입니다."
            prompt = generate_prompt(question, answer)
            data_list.append(prompt)
            
            # 3. 음식점의 휴무일 질문
            question = f"{restaurant_name}의 휴무일은 언제인가요?"
            answer = f"{restaurant_name}의 휴무일은 {holiday}입니다."
            prompt = generate_prompt(question, answer)
            data_list.append(prompt)
            
            # 4. 주차시설 유무 질문
            question = f"{restaurant_name}에 주차시설이 있나요?"
            answer = f"{restaurant_name}은 주차시설이 {parking}."
            prompt = generate_prompt(question, answer)
            data_list.append(prompt)
            
            # 5. 대표 메뉴 질문
            question = f"{restaurant_name}의 대표 메뉴는 무엇인가요?"
            answer = f"{restaurant_name}의 대표 메뉴는 {menu}입니다."
            prompt = generate_prompt(question, answer)
            data_list.append(prompt)
            
    return data_list
            

def preprocess_data(data_list, tokenizer):
    dataset = CustomDataset(data_list, tokenizer)
    return dataset
