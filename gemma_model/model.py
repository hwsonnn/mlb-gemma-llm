from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b",
                attn_implementation='eager'
            )  # 'eager' 어텐션 구현 사용
    return tokenizer, model

def get_finetuned_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("../fine_tuned_model")
    model = AutoModelForCausalLM.from_pretrained("../fine_tuned_model")
    return tokenizer, model