from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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
                quantization_config=bnb_config
            )  

    return tokenizer, model


def save_model(trainer, tokenizer):
    BASE_MODEL = "google/gemma-2-2b-it"
    ADAPTER_MODEL = "../lora_adapter"
    
    trainer.model.save_pretrained(ADAPTER_MODEL)
    tokenizer.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained('../fine_tuned_model')


def get_finetuned_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("../fine_tuned_model")
    model = AutoModelForCausalLM.from_pretrained("../fine_tuned_model")
    return tokenizer, model
