from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline

import torch


def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    tokenizer.padding_side = 'right'
    special_tokens = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'additional_special_tokens': ['<start_of_turn>user', '<start_of_turn>model', '<end_of_turn>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
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

    return tokenizer, model


def save_model(trainer, tokenizer):
    BASE_MODEL = "google/gemma-2-2b-it"
    ADAPTER_MODEL = "../lora_adapter"
    FINETUNED_MODEL = '../fine_tuned_model'
    
    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained(FINETUNED_MODEL)
    tokenizer.save_pretrained(FINETUNED_MODEL)
    
    return True


def get_finetuned_pipeline():
    FINETUNE_MODEL = "../fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})

    return pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)
