from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b",
                attn_implementation='eager', # 'eager' 어텐션 구현 사용
                load_in_8bit=True # 양자화 적용 (8비트)
            )

    # LoRA 설정 - 메모리 부족 현상 해결을 위함
    lora_config = LoraConfig(
        r=8,  # Low-rank matrix의 rank 값 (작을수록 메모리 절약)
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # 올바른 모듈 이름으로 변경
        lora_dropout=0.1,
        bias="none"
    )

    # LoRA 적용된 모델 생성
    lora_model = get_peft_model(model, lora_config)

    # LoRA 파라미터만 학습 가능하도록 설정
    for name, param in lora_model.named_parameters():
      # LoRA 파라미터만 requires_grad=True로 설정 (업데이트할 파라미터만 학습)
      if 'lora_' in name:  # LoRA로 추가된 파라미터만 학습
        # print(name, param)
        param.requires_grad = True
      else:
        param.requires_grad = False  # 기본적으로 모든 파라미터를 학습하지 않도록 설정

    return tokenizer, lora_model

def get_finetuned_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("../fine_tuned_model")
    model = AutoModelForCausalLM.from_pretrained("../fine_tuned_model")
    return tokenizer, model