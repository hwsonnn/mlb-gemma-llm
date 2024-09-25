from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
from trl import setup_chat_format

import bitsandbytes as bnb
import torch

BASE_MODEL = "google/gemma-2-2b-it"
ADAPTER_MODEL = "../lora_adapter"
FINETUNED_MODEL = "../gemma-2-2b-it-lora-fine-tuned"


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_tokenizer_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                attn_implementation='eager', # 'eager' 어텐션 구현 사용
                device_map="auto",
                quantization_config=bnb_config
            )  
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    # LoRA 설정
    modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=8,  # Low-rank matrix의 rank 값 (작을수록 메모리 절약)
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, lora_config)

    return model, tokenizer, lora_config


def save_model(trainer, tokenizer):
    # 1. 어댑터 모델 저장
    trainer.model.save_pretrained(ADAPTER_MODEL)

    # 2. 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', return_dict=True, torch_dtype=torch.float16)
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)

    # 3. 어댑터 모델 로드
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

    # 4. LoRA 어댑터 병합 및 언로드
    model = model.merge_and_unload()

    # 5. 모델과 토크나이저 저장
    model.save_pretrained(FINETUNED_MODEL)
    tokenizer.save_pretrained(FINETUNED_MODEL)
    
    return True


def get_finetuned_pipeline():
    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    return pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)
