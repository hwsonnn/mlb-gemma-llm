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


def generate_qa_pairs(df):
    questions = []
    answers = []
    
    for _, row in df.iterrows():
        restaurant_name = row['음식점명']
        
        # 1. 메뉴 질문
        questions.append(f"{restaurant_name}의 대표 메뉴는 뭐야?")
        answers.append(f"대표 메뉴는 {row['취급메뉴']}입니다.")
        
        # 2. 주소 및 추천 질문
        questions.append(f"{' '.join(row['주소'].split()[:3])} 근처에 있는 {row['취급메뉴']} 맛집 추천해줘")
        answers.append(f"{row['음식점명']}을 추천합니다. 주소는 {row['주소']}이고, 특징은 다음과 같습니다:\n\
                        - {row['개요']}\n\
                        - 리뷰: {row['리뷰키워드']}\n\
                        - 주차시설: {row['주차시설 유무']}")
        
        # 3. 영업시간 질문
        questions.append(f"{restaurant_name}의 영업시간은 어떻게 돼?")
        answers.append(f"{restaurant_name}의 영업시간은 {row['영업시간']}입니다.")
        
        # 4. 휴무일 질문
        questions.append(f"{restaurant_name}의 휴무일은 언제야?")
        answers.append(f"휴무일은 {row['휴무일']}입니다.")
    
    return questions, answers


def load_data(file_path: str) -> list:
    total_restaurant_info = []
    total_questions = []
    total_answers = []
    
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            tmp_df = pd.read_csv(file_path + file)
            
            restaurant_info = ', '.join(f'{row.iloc[0]}: "{row.iloc[1]}"' for _, row in tmp_df.iterrows())
            total_restaurant_info.append(restaurant_info)
            
            questions, answers = generate_qa_pairs(tmp_df)
            total_questions.extend(questions)
            total_answers.extend(answers)
            
    return total_restaurant_info, total_questions, total_answers
            

def preprocess_data(restaurant_info: list[str], questions: list[str], answers: list[str], tokenizer) -> CustomDataset:
    combined_inputs = [f'{info} {q}' for info, q in zip(restaurant_info, questions)]
    inputs = tokenizer(combined_inputs, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
    
    # labels 추가 
    combined_labels = [f'{info} {a}' for info, a in zip(restaurant_info, answers)]
    labels = tokenizer(combined_labels, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')['input_ids']

    # # (input_ids를 복사하여 사용) = 모델이 내재적인 지식으로 학습할 수 있도록 구성
    # # encodings['labels'] = encodings['input_ids'].clone()
    # 질문/답변 데이터를 추가해서 각각 input/labels 로 활용
    inputs['labels'] = labels
    
    return CustomDataset(inputs)