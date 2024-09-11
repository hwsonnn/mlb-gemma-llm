import pandas as pd
from transformers import Trainer, TrainingArguments

from .data_loader import load_data, preprocess_data
from .model import get_tokenizer_model

# 1. CSV 데이터 로드
train_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Training/01_raw_data/TS_triple_jeju_food/'
eval_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Validation/01_raw_data/VS_triple_jeju_food/'
train_data = load_data(train_file_path)
eval_data = load_data(eval_file_path)

# 2. 토크나이저 및 모델 로드 (Gemma 모델 활용)
tokenizer, model = get_tokenizer_model()

# 3. 데이터 전처리 (토크나이저로 텍스트 데이터 토큰화)
train_encodings = preprocess_data(train_data, tokenizer)
eval_encodings = preprocess_data(eval_data, tokenizer)

# 4. 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# 5. Trainer 설정 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,  # train 데이터 설정
    eval_dataset=eval_encodings,    # validation 데이터 설정
    tokenizer=tokenizer,
)

# 6. 모델 훈련
trainer.train()
