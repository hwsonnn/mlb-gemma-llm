from transformers import Trainer, TrainingArguments

from data_loader import load_data, preprocess_data
from model import get_tokenizer_model

import os
import torch

# MPS 사용 비활성화를 위한 환경 변수 설정 # GPU 사용시 주석 처리 필요
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 1. CSV 데이터 로드
train_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Training/01_raw_data/TS_triple_jeju_food/'
eval_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Validation/01_raw_data/VS_triple_jeju_food/'
train_restaurant_info, train_questions, train_answers = load_data(train_file_path)
eval_restaurant_info, eval_questions, eval_answers = load_data(eval_file_path)

# 2. 토크나이저 및 모델 로드 (Gemma 모델 활용)
tokenizer, model = get_tokenizer_model()

# 3. 데이터 전처리 (토크나이저로 텍스트 데이터 토큰화)
train_dataset = preprocess_data(train_restaurant_info, train_questions, train_answers, tokenizer)
eval_dataset = preprocess_data(eval_restaurant_info, eval_questions, eval_answers, tokenizer)

# 4. 훈련 설정
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # MPS가 가능하면 MPS, 아니면 CPU
# device = torch.device("cpu")  # MPS가 가능하면 MPS, 아니면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device=="cuda": torch.cuda.empty_cache()
model.to(device)
print(f"모델이 사용 중인 디바이스: {device}")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2, # MPS allocated 문제: 12로 하니까 에러 발생
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # 기울기 누적을 통해 4배의 배치 크기 효과
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Mixed Precision 활성화
    # gradient_checkpointing=True  # Checkpointing 활성화
)

# 5. Trainer 설정 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # train 데이터 설정
    eval_dataset=eval_dataset,    # validation 데이터 설정
    tokenizer=tokenizer,
)

# 6. 모델 훈련
trainer.train()
# trainer.train(resume_from_checkpoint=True)
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
