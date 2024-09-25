from transformers import TrainingArguments
from trl import SFTTrainer

from data_loader import load_data, preprocess_data, generate_prompt
from model import get_tokenizer_model, save_model

import os

# MPS 사용 비활성화를 위한 환경 변수 설정 # GPU 사용시 주석 처리 필요
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 1. CSV 데이터 로드
train_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Training/01_raw_data/TS_triple_jeju_food/'
eval_file_path = '../data/219_2_KVQA_jeju_tour_data/01_1_official_public_data/Validation/01_raw_data/VS_triple_jeju_food/'

# 2. 토크나이저 및 모델 로드 (Gemma 모델 활용)
model, tokenizer, lora_config = get_tokenizer_model()

# 3. 데이터 전처리 (토크나이저로 텍스트 데이터 토큰화)
train_dataset = preprocess_data(load_data(train_file_path), tokenizer)

# 4. 훈련 설정
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # MPS가 가능하면 MPS, 아니면 CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device=="cuda": torch.cuda.empty_cache()
# model.to(device)
# print(f"모델이 사용 중인 디바이스: {device}")

# 5. Trainer 설정 및 훈련
# Trainer 클래스 사용 X - https://devocean.sk.com/blog/techBoardDetail.do?ID=165703&boardType=techBlog
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset["train"],
    eval_dataset=train_dataset["test"],
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
        num_train_epochs = 3,
        max_steps=3000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        learning_rate=2e-4,
        fp16=False,             # 중요
        bf16=False,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
    ),
    peft_config=lora_config,
    dataset_text_field="text",
)

# 6. 모델 훈련
trainer.train()

# 7. 모델 저장
save_model(trainer, tokenizer)
